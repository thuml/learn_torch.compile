
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


# kernel path: /tmp/torchinductor_youkaichao/kk/ckk2la3fadvebn55sajyrigxz33k5pflftkiqd3ewetpdgudecij.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]

triton_red_fused_native_layer_norm_backward_select_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_select_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1576
    rnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 197
    x1 = (xindex // 197)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp3 = tl.load(in_ptr0 + (r2 + (384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp7 = tmp5 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp12 = tmp7 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp16 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp20 = tl.load(in_ptr0 + (r2 + (384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr2 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = x0
        tmp18 = tl.full([1, 1], 0, tl.int32)
        tmp19 = tmp17 == tmp18
        tmp21 = 0.0
        tmp22 = tl.where(tmp19, tmp20, tmp21)
        tmp24 = tmp22 * tmp23
        tmp25 = 384.0
        tmp26 = tmp24 * tmp25
        tmp27 = tmp26 - tmp9
        tmp29 = tmp28 * tmp14
        tmp30 = tmp27 - tmp29
        tmp31 = tmp16 * tmp30
        tl.store(out_ptr2 + (r2 + (384*x3)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oq/coq3ryvkpyvc7tdosk2qnsjjyth2edyo5do37bezupntlt7hgdl3.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]

triton_red_fused_native_layer_norm_backward_select_backward_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_select_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 122
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
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = (r2 + (122*x1)) % 197
        tmp4 = tl.full([1, 1], 0, tl.int32)
        tmp5 = tmp3 == tmp4
        tmp6 = tl.load(in_ptr0 + (x0 + (384*(((r2 + (122*x1)) // 197) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = 0.0
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tl.load(in_ptr1 + (x0 + (384*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tmp8 * tmp9
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/za/czacf23akk5nmmw4q3uhhr5ymdtn7pa4hjq24jxrwt7wfiw5oido.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]

triton_per_fused_native_layer_norm_backward_select_backward_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_select_backward_3', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3g/c3goieowpm7iqhk45un6uaatbfhk6ihkxtp2xdf72ov542fbzsfk.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]

triton_red_fused_native_layer_norm_backward_select_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_select_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 1576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 197
        r2 = (rindex // 197)
        tmp3 = tl.load(in_ptr0 + (x0 + (384*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = r1
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f2/cf25kt7bmz3ilyhhjqrtbzc3wbluenr33wq4hxucb56gr67h4732.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*r2) + (46848*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wu/cwuxmnbprcizltitqfpfsca7zcmxy322oolbuf7dx5mtk5tt2fr4.py
# Source Nodes: [x_216], Original ATen: [aten.gelu, aten.gelu_backward]
# x_216 => add_210, erf_23, mul_218
triton_poi_fused_gelu_gelu_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2420736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gc/cgc5rkin2s3olkq4t5u6j6nfqsbin2gkmoqeyqeaovekbzg3hfow.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 19968
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 1536)
    x0 = xindex % 1536
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (1536*r2) + (187392*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/em/cempzdhbav2v6q5rzkzjfu4uosxdybqulbsjljo3il465b5efpcs.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1536*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hv/chvogb5mabd76lwmc7rd5yxaxwsstyl7ymytq3kpg7xrv2pphpjs.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 384.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (384*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e3/ce3orsfecgil3uzz72cpvv6s2ive5fbrbwa3y2yb4q2c6g2ajro6.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (384*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp11 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp12 = tl.where(tmp2, tmp3, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4w/c4wa4xkfluo2w47ehgobudhu4emmalrs7r4be6p4hm7ws4hpwgaz.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 197
    x2 = (xindex // 12608) % 6
    x3 = (xindex // 75648)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (384*x1) + (75648*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/af/cafy5zeq5bpxouh3oian3zpqq4jg66ndrfz5qlrukm5wfhgghafp.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]

triton_per_fused__softmax_backward_data_mul_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 9456
    rnumel = 197
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (197*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (197*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr1 + (r1 + (197*x0)), tmp10, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wy/cwyux3avbd6hc4hk36f3ixkfkjro7axsybwfdv523yuzug6rd4ca.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 197)) + (12608*(x0 // 64)) + (75648*(x1 // 197)) + (x0 % 64)), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/43/c43tudvnbj2hthbult32mbjdmdqgw5odo4mmzppyygrhvtst6va6.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 18912
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y6 = (yindex // 1182)
    x4 = xindex
    y7 = yindex
    y0 = yindex % 197
    y8 = (yindex // 197)
    y1 = (yindex // 197) % 6
    y2 = (yindex // 1182) % 8
    y3 = (yindex // 9456)
    tmp0 = y6
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + (64*y7)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 16, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-605184) + y0 + (197*x4) + (12608*y8)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x4 + (64*y1) + (384*y3) + (768*y0) + (151296*y2)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/on/con4tuhft2xpiuhd3brwuwd3luimdoji2fmgib74zddg55c22zcy.py
# Source Nodes: [l__mod___blocks_11_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___blocks_11_norm_out => mul_212, sub_83
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_out_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp11 = tmp9 - tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = tmp4 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 384.0
    tmp21 = tmp12 / tmp20
    tmp22 = tmp4 * tmp20
    tmp23 = tmp22 - tmp8
    tmp24 = tmp13 * tmp18
    tmp25 = tmp23 - tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp19 + tmp26
    tl.store(in_out_ptr0 + (r1 + (384*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ch/cchnty24jl4erit6xxbigoub6txiske3t7xs6sbqcixgm6pilw7w.py
# Source Nodes: [l__mod___blocks_11_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___blocks_11_norm_out => mul_212, sub_83
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (384*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.load(in_ptr2 + (x0 + (384*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr3 + ((r2 + (122*x1)) % 1576), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 - tmp7
        tmp9 = tl.load(in_ptr4 + ((r2 + (122*x1)) % 1576), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 * tmp9
        tmp11 = tmp5 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
        tmp17 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp18 = tl.where(tmp2, tmp5, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mo/cmozbkq2mn4rtg5uxrfcqqb7hwy5h4cm5ixkfw3d7rk6aapoefxy.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]

triton_poi_fused__unsafe_view_clone_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (384 + x0 + (384*(x1 % 196)) + (75648*(x1 // 196))), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fg/cfg2sujrzdmgbpcljv6w7qop645ep2i2jmxuuyiojsv4lp24zdea.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*r2) + (46464*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dg/cdgq3wbjjfrd5ommll7oicjb26eisce3qtl5guhluyty6zml24yq.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = 24.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tl.store(out_ptr2 + (r1 + (24*x0)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hi/chiuscocaeix376lihawhxfb7phgl44kupuh4edwkyz55cr7tl7t.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (24*r2) + (3072*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (24*r2) + (3072*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jh/cjhaz3g4s5336reqicuuqgzpagkaikhfseaz25mkqnl6vl6xgirj.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24
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
        tmp0 = tl.load(in_ptr0 + (x0 + (24*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7y/c7yxz6j6cx6q6rnvkvjivvyhope2f53ymrzlmtq5jgdjdqaa2czn.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (24*r2) + (3072*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6g/c6gt4ovyenlmqpcggns6hq4cdoxgfm2os2ms2q4k4vbws2akxzxk.py
# Source Nodes: [x_207], Original ATen: [aten.gelu, aten.gelu_backward]
# x_207 => add_200, erf_22, mul_208
triton_poi_fused_gelu_gelu_backward_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zt/cztqfm5whu6x3xc5j7k4okcen5iyhfl4cyq6l7hnadbjmeqjlnzd.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18816
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 96
    x1 = (xindex // 96)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2m/c2m2tcdgrj2ggpruzss3mgwiirgboyemo42ihb5yrebfstyzftkl.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/3w/c3wrajuswtkb2ezhoqxqazamypqqlmimw7xcicnjxnyhvappb3ql.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_26', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = 24.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (24*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kq/ckqs44vjunokhen4bqijoukdl6wyvkc3a4xfftwenxc7so3mm7fc.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = (xindex // 6) % 16
    x2 = (xindex // 96) % 4
    x3 = (xindex // 384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (6*x2) + (24*x1) + (384*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mt/cmtrm5wnucac6a4aq7opo6iwgxeojco4nydsa5oabw24ag3igi6x.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]

triton_per_fused__softmax_backward_data_mul_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (16*x0)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tmp9 = 0.408248290463863
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr1 + (r1 + (16*x0)), tmp10, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s6/cs66ponjl7unj4y4un56edjyqngt4a6jmxl6skf3yviyhghldoo3.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 24
    x1 = (xindex // 24)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((6*(x1 % 16)) + (96*(x0 // 6)) + (384*(x1 // 16)) + (x0 % 6)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3n/c3n2wt7q4puvmxycvv6rpuajkxcb4cjmvkzgvrvwlmxcpo7krecc.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 8], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 200704
    xnumel = 6
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y6 = (yindex // 64)
    x4 = xindex
    y7 = yindex
    y0 = yindex % 16
    y8 = (yindex // 16)
    y1 = (yindex // 16) % 4
    y2 = (yindex // 64) % 1568
    y3 = (yindex // 100352)
    tmp0 = y6
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1568, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + (6*y7)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 3136, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-602112) + y0 + (16*x4) + (96*y8)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x4 + (6*y1) + (24*y3) + (48*y0) + (768*y2)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zq/czqhe73f3qdcpujnpe6lx4e5umohqrieykv5bfcarlz2bf5pmqo6.py
# Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward]

triton_poi_fused_add_select_backward_slice_backward_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_select_backward_slice_backward_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 384) % 197
    x3 = xindex
    x0 = xindex % 384
    x2 = (xindex // 75648)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x3), tmp2 & xmask, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr0 + (x0 + (75648*x2)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e3/ce3fd3dlth66pw45g3wztkyuwynz2tg7rojacysh4r57mt3jlcrp.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 384.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rk/crkuxt62pexpkhhrvpusoq4ksi6yvqvoubj33q4kfsxzmrn3sja5.py
# Source Nodes: [l__mod___blocks_10_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___blocks_10_norm_out => mul_194, sub_76
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp11 = tmp9 - tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = tmp4 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 384.0
    tmp21 = tmp12 / tmp20
    tmp22 = tmp4 * tmp20
    tmp23 = tmp22 - tmp8
    tmp24 = tmp13 * tmp18
    tmp25 = tmp23 - tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp19 + tmp26
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kt/ckttsmonfeyopucjem54dwucznwz4vav2ezmng37lnd64rq6ztpa.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_34', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_ptr4 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.load(in_out_ptr0 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp17 = tmp15 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = tmp17 * tmp9
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp29 = 24.0
    tmp30 = tmp4 * tmp29
    tmp31 = tmp30 - tmp8
    tmp32 = tmp9 * tmp14
    tmp33 = tmp31 - tmp32
    tmp34 = tmp28 * tmp33
    tmp35 = tmp27 + tmp34
    tmp36 = tmp17 * tmp29
    tmp37 = tmp36 - tmp21
    tmp38 = tmp9 * tmp26
    tmp39 = tmp37 - tmp38
    tmp40 = tmp28 * tmp39
    tmp41 = tmp35 + tmp40
    tl.store(in_out_ptr0 + (r1 + (24*x0)), tmp41, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ob/cobwajkp2swlmjckvyz4fczhr5gjw57nfsonn5cxqpgpth2usvn5.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (24*r2) + (3072*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (24*r2) + (3072*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (24*r2) + (3072*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (x0 + (24*r2) + (3072*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp8 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp12 = tmp11 * tmp3
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
        tmp16 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp14, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr3 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mm/cmmdopb4fkvranxaczcyyclmasksgjqidu5uia6iutu6mgdotp6f.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*r2) + (46848*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kg/ckgeguh3v2bmplybc2dfuecr7tqvbl6xgqxh2sjsbkhfcxsczqv7.py
# Source Nodes: [patch_embed], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# patch_embed => mul_2, sub_1
triton_per_fused_native_layer_norm_native_layer_norm_backward_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 196
    r2 = rindex
    x1 = (xindex // 196)
    x3 = xindex
    tmp14 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr2 + (r2 + (384*x3)), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = 1 + x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (384 + r2 + (384*x0) + (75648*x1)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr0 + (r2 + (75648*x1)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp22 = tmp20 - tmp21
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = 384.0
    tmp31 = tmp23 / tmp30
    tmp32 = tmp15 * tmp30
    tmp33 = tmp32 - tmp19
    tmp34 = tmp24 * tmp29
    tmp35 = tmp33 - tmp34
    tmp36 = tmp31 * tmp35
    tl.store(out_ptr2 + (r2 + (384*x3)), tmp36, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2b/c2bpfwmjho3gfcef7t72iwlovnuanha4nyt226vswcxx377zcit5.py
# Source Nodes: [l__mod___norm1_proj], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___norm1_proj => mul, sub
triton_per_fused_native_layer_norm_native_layer_norm_backward_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp9 = tmp7 - tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tmp2 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t5/ct5r2sv7qqhnwzqk7fxcf2mplkfl46zlbupww2iqfrsvy2h5zz5a.py
# Source Nodes: [l__mod___blocks_0_norm_in], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___blocks_0_norm_in => mul_4, sub_2
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_39', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x3 = (xindex // 16)
    x2 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_out_ptr0 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp32 = tl.load(in_ptr8 + (r1 + (24*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.load(in_ptr9 + (x3), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr10 + (x3), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr11 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp11 = tmp9 - tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = tmp4 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = 24.0
    tmp21 = tmp12 / tmp20
    tmp22 = tmp4 * tmp20
    tmp23 = tmp22 - tmp8
    tmp24 = tmp13 * tmp18
    tmp25 = tmp23 - tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp19 + tmp26
    tmp29 = 384.0
    tmp30 = tmp28 / tmp29
    tmp33 = tmp31 * tmp32
    tmp34 = tmp33 * tmp29
    tmp36 = tmp34 - tmp35
    tmp38 = tmp9 - tmp37
    tmp39 = tmp38 * tmp28
    tmp41 = tmp39 * tmp40
    tmp42 = tmp36 - tmp41
    tmp43 = tmp30 * tmp42
    tmp44 = tmp27 + tmp43
    tl.store(in_out_ptr0 + (r1 + (24*x0)), tmp44, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pw/cpwcti2mad5y7imji4wssumeqdz6hmnn54jaaxv35x4rnzaizf66.py
# Source Nodes: [l__mod___blocks_0_norm_in], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___blocks_0_norm_in => mul_4, sub_2
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (24*r2) + (3072*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (24*r2) + (3072*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (24*r2) + (3072*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r2 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr4 + (r2 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 - tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp12 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cz/cczydfrevmicess5r26lek2nand77b6nnzzksfc5nsuziz4ctlyx.py
# Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward, aten.sum]

triton_per_fused_add_select_backward_slice_backward_sum_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_select_backward_slice_backward_sum_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 75648
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x1 = (xindex // 384)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 384
    tmp0 = x1
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x3 + (75648*r2)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr0 + (x0 + (75648*r2)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ic/cic5jco4c5n63g4y2wltdrhxcmblerdrzwcpptlkcwdteofm3isi.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.full([1, 1], 0, tl.int64)
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + (75648*r1)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr0 + (x0 + (75648*r1)), rmask & tmp8 & xmask, other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dn/cdnnhro5oxvu4s7g7tubvwgjapsk2e7giehv33s35bqes7jp7czi.py
# Source Nodes: [patch_embed], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# patch_embed => mul_2, sub_1
triton_red_fused_native_layer_norm_native_layer_norm_backward_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp33 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = 1 + ((r2 + (121*x1)) % 196)
        tmp4 = tl.full([1, 1], 1, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tmp5 & tmp2
        tmp7 = tl.load(in_ptr0 + (384 + x0 + (384*((r2 + (121*x1)) % 196)) + (75648*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp6 & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp10 = 0.0
        tmp11 = tl.where(tmp5, tmp9, tmp10)
        tmp12 = tmp3 < tmp4
        tmp13 = tmp12 & tmp2
        tmp14 = tl.load(in_ptr0 + (x0 + (75648*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp13 & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
        tmp16 = tl.where(tmp13, tmp14, tmp15)
        tmp17 = tl.where(tmp12, tmp16, tmp10)
        tmp18 = tmp11 + tmp17
        tmp19 = tl.load(in_ptr1 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr2 + ((r2 + (121*x1)) % 1568), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp19 - tmp20
        tmp22 = tl.load(in_ptr3 + ((r2 + (121*x1)) % 1568), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tmp21 * tmp22
        tmp24 = tmp18 * tmp23
        tmp25 = tl.full(tmp24.shape, 0, tmp24.dtype)
        tmp26 = tl.where(tmp2, tmp24, tmp25)
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
        tmp30 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp31 = tl.where(tmp2, tmp18, tmp30)
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp34 = _tmp33 + tmp32
        _tmp33 = tl.where(rmask & xmask, tmp34, _tmp33)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp33, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4j/c4jcis5xqm5vysyesngoipin4jqu2zw7vamhixvtjyajlmkgvsdz.py
# Source Nodes: [l__mod___norm1_proj], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___norm1_proj => mul, sub
triton_red_fused_native_layer_norm_native_layer_norm_backward_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + ((r2 + (121*x1)) % 1568), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tl.load(in_ptr3 + ((r2 + (121*x1)) % 1568), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp3 * tmp8
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp15 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp16 = tl.where(tmp2, tmp3, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6r/c6rcy77sxgg6anse5z3z5b2nostdrkdqgw4yw7uqubgsqqidszma.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*r2) + (46464*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k6/ck6ok6rl3hylfghrf2zfr6uumddr4qgqt4o3hb6p57dzqeeftlxi.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_46', 'mutated_arg_names': []}
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
    r2 = rindex
    x0 = xindex % 16
    x1 = (xindex // 16)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + (24*x0) + (384*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/27/c27euosbjjklxnf47nij4uxnlgyqbiwibz2vfborf25epjaryzsl.py
# Source Nodes: [], Original ATen: [aten.col2im]

triton_poi_fused_col2im_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_col2im_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zu/czuf355s3akcoxzi23xtqhvelm3jf7mufyjcrvhyge5nfyluvt7i.py
# Source Nodes: [], Original ATen: [aten.clone, aten.col2im]

triton_poi_fused_clone_col2im_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_col2im_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 37632
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (24*x2) + (384*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (16*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pg/cpgug6634arfpa4bdaifgmi3grdfvaqkrzq6p5hnoswinzlbsbb5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_49', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/yo/cyoxuz4u75xwx6fbz4nrhsgrhdji7at7hl7qntsxw7zmmsiornbv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_50', 'mutated_arg_names': []}
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_4, primals_6, primals_10, primals_12, primals_18, primals_24, primals_28, primals_34, primals_40, primals_46, primals_52, primals_56, primals_62, primals_68, primals_74, primals_80, primals_84, primals_90, primals_96, primals_102, primals_108, primals_112, primals_118, primals_124, primals_130, primals_136, primals_140, primals_146, primals_152, primals_158, primals_164, primals_168, primals_174, primals_180, primals_186, primals_192, primals_196, primals_202, primals_208, primals_214, primals_220, primals_224, primals_230, primals_236, primals_242, primals_248, primals_252, primals_258, primals_264, primals_270, primals_276, primals_280, primals_286, primals_292, primals_298, primals_304, primals_308, primals_314, primals_320, primals_326, primals_332, primals_336, primals_342, primals_348, primals_352, add, unsqueeze_5, clone_2, getitem_1, rsqrt, view_4, addmm, getitem_3, rsqrt_1, getitem_5, rsqrt_2, view_6, view_19, mul_7, view_21, addmm_2, view_23, mul_12, view_26, cat_1, getitem_13, rsqrt_5, view_28, view_41, mul_17, view_43, addmm_6, view_45, view_47, view_60, mul_25, view_62, addmm_9, view_64, mul_30, view_67, cat_2, getitem_27, rsqrt_10, view_69, view_82, mul_35, view_84, addmm_13, view_86, view_88, view_101, mul_43, view_103, addmm_16, view_105, mul_48, view_108, cat_3, getitem_41, rsqrt_15, view_110, view_123, mul_53, view_125, addmm_20, view_127, view_129, view_142, mul_61, view_144, addmm_23, view_146, mul_66, view_149, cat_4, getitem_55, rsqrt_20, view_151, view_164, mul_71, view_166, addmm_27, view_168, view_170, view_183, mul_79, view_185, addmm_30, view_187, mul_84, view_190, cat_5, getitem_69, rsqrt_25, view_192, view_205, mul_89, view_207, addmm_34, view_209, view_211, view_224, mul_97, view_226, addmm_37, view_228, mul_102, view_231, cat_6, getitem_83, rsqrt_30, view_233, view_246, mul_107, view_248, addmm_41, view_250, view_252, view_265, mul_115, view_267, addmm_44, view_269, mul_120, view_272, cat_7, getitem_97, rsqrt_35, view_274, view_287, mul_125, view_289, addmm_48, view_291, view_293, view_306, mul_133, view_308, addmm_51, view_310, mul_138, view_313, cat_8, getitem_111, rsqrt_40, view_315, view_328, mul_143, view_330, addmm_55, view_332, view_334, view_347, mul_151, view_349, addmm_58, view_351, mul_156, view_354, cat_9, getitem_125, rsqrt_45, view_356, view_369, mul_161, view_371, addmm_62, view_373, view_375, view_388, mul_169, view_390, addmm_65, view_392, mul_174, view_395, cat_10, getitem_139, rsqrt_50, view_397, view_410, mul_179, view_412, addmm_69, view_414, view_416, view_429, mul_187, view_431, addmm_72, view_433, mul_192, view_436, cat_11, getitem_153, rsqrt_55, view_438, view_451, mul_197, view_453, addmm_76, view_455, view_457, view_470, mul_205, view_472, addmm_79, view_474, mul_210, view_477, cat_12, getitem_167, rsqrt_60, view_479, view_492, mul_215, view_494, addmm_83, view_496, mul_220, clone_184, permute_233, div_24, permute_237, permute_241, div_25, permute_245, permute_250, permute_251, alias_24, permute_252, permute_253, permute_258, permute_263, permute_265, div_27, permute_269, permute_273, div_28, permute_277, permute_282, permute_283, alias_25, permute_284, permute_285, permute_290, permute_295, div_29, permute_297, permute_301, div_30, permute_305, permute_310, permute_311, alias_26, permute_312, permute_313, permute_318, permute_323, permute_325, permute_329, permute_333, div_33, permute_337, permute_342, permute_343, alias_27, permute_344, permute_345, permute_350, permute_355, div_34, permute_357, permute_361, div_35, permute_365, permute_370, permute_371, alias_28, permute_372, permute_373, permute_378, permute_383, permute_385, permute_389, permute_393, div_38, permute_397, permute_402, permute_403, alias_29, permute_404, permute_405, permute_410, permute_415, div_39, permute_417, permute_421, div_40, permute_425, permute_430, permute_431, alias_30, permute_432, permute_433, permute_438, permute_443, permute_445, permute_449, permute_453, div_43, permute_457, permute_462, permute_463, alias_31, permute_464, permute_465, permute_470, permute_475, div_44, permute_477, permute_481, div_45, permute_485, permute_490, permute_491, alias_32, permute_492, permute_493, permute_498, permute_503, permute_505, permute_509, permute_513, div_48, permute_517, permute_522, permute_523, alias_33, permute_524, permute_525, permute_530, permute_535, div_49, permute_537, permute_541, div_50, permute_545, permute_550, permute_551, alias_34, permute_552, permute_553, permute_558, permute_563, permute_565, permute_569, permute_573, div_53, permute_577, permute_582, permute_583, alias_35, permute_584, permute_585, permute_590, permute_595, div_54, permute_597, permute_601, div_55, permute_605, permute_610, permute_611, alias_36, permute_612, permute_613, permute_618, permute_623, permute_625, permute_629, permute_633, div_58, permute_637, permute_642, permute_643, alias_37, permute_644, permute_645, permute_650, permute_655, div_59, permute_657, permute_661, div_60, permute_665, permute_670, permute_671, alias_38, permute_672, permute_673, permute_678, permute_683, permute_685, permute_689, permute_693, div_63, permute_697, permute_702, permute_703, alias_39, permute_704, permute_705, permute_710, permute_715, div_64, permute_717, permute_721, div_65, permute_725, permute_730, permute_731, alias_40, permute_732, permute_733, permute_738, permute_743, permute_745, permute_749, permute_753, div_68, permute_757, permute_762, permute_763, alias_41, permute_764, permute_765, permute_770, permute_775, div_69, permute_777, permute_781, div_70, permute_785, permute_790, permute_791, alias_42, permute_792, permute_793, permute_798, permute_803, permute_805, permute_809, permute_813, div_73, permute_817, permute_822, permute_823, alias_43, permute_824, permute_825, permute_830, permute_835, div_74, permute_837, permute_841, div_75, permute_845, permute_850, permute_851, alias_44, permute_852, permute_853, permute_858, permute_863, permute_865, permute_869, permute_873, div_78, permute_877, permute_882, permute_883, alias_45, permute_884, permute_885, permute_890, permute_895, div_79, permute_897, permute_901, div_80, permute_905, permute_910, permute_911, alias_46, permute_912, permute_913, permute_918, permute_923, permute_925, permute_929, permute_933, div_83, permute_937, permute_942, permute_943, alias_47, permute_944, permute_945, permute_950, permute_955, permute_957, tangents_1 = args
    args.clear()
    assert_size_stride(primals_4, (24, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_6, (384, ), (1, ))
    assert_size_stride(primals_10, (384, ), (1, ))
    assert_size_stride(primals_12, (24, ), (1, ))
    assert_size_stride(primals_18, (24, ), (1, ))
    assert_size_stride(primals_24, (24, ), (1, ))
    assert_size_stride(primals_28, (384, ), (1, ))
    assert_size_stride(primals_34, (384, ), (1, ))
    assert_size_stride(primals_40, (24, ), (1, ))
    assert_size_stride(primals_46, (24, ), (1, ))
    assert_size_stride(primals_52, (24, ), (1, ))
    assert_size_stride(primals_56, (384, ), (1, ))
    assert_size_stride(primals_62, (384, ), (1, ))
    assert_size_stride(primals_68, (24, ), (1, ))
    assert_size_stride(primals_74, (24, ), (1, ))
    assert_size_stride(primals_80, (24, ), (1, ))
    assert_size_stride(primals_84, (384, ), (1, ))
    assert_size_stride(primals_90, (384, ), (1, ))
    assert_size_stride(primals_96, (24, ), (1, ))
    assert_size_stride(primals_102, (24, ), (1, ))
    assert_size_stride(primals_108, (24, ), (1, ))
    assert_size_stride(primals_112, (384, ), (1, ))
    assert_size_stride(primals_118, (384, ), (1, ))
    assert_size_stride(primals_124, (24, ), (1, ))
    assert_size_stride(primals_130, (24, ), (1, ))
    assert_size_stride(primals_136, (24, ), (1, ))
    assert_size_stride(primals_140, (384, ), (1, ))
    assert_size_stride(primals_146, (384, ), (1, ))
    assert_size_stride(primals_152, (24, ), (1, ))
    assert_size_stride(primals_158, (24, ), (1, ))
    assert_size_stride(primals_164, (24, ), (1, ))
    assert_size_stride(primals_168, (384, ), (1, ))
    assert_size_stride(primals_174, (384, ), (1, ))
    assert_size_stride(primals_180, (24, ), (1, ))
    assert_size_stride(primals_186, (24, ), (1, ))
    assert_size_stride(primals_192, (24, ), (1, ))
    assert_size_stride(primals_196, (384, ), (1, ))
    assert_size_stride(primals_202, (384, ), (1, ))
    assert_size_stride(primals_208, (24, ), (1, ))
    assert_size_stride(primals_214, (24, ), (1, ))
    assert_size_stride(primals_220, (24, ), (1, ))
    assert_size_stride(primals_224, (384, ), (1, ))
    assert_size_stride(primals_230, (384, ), (1, ))
    assert_size_stride(primals_236, (24, ), (1, ))
    assert_size_stride(primals_242, (24, ), (1, ))
    assert_size_stride(primals_248, (24, ), (1, ))
    assert_size_stride(primals_252, (384, ), (1, ))
    assert_size_stride(primals_258, (384, ), (1, ))
    assert_size_stride(primals_264, (24, ), (1, ))
    assert_size_stride(primals_270, (24, ), (1, ))
    assert_size_stride(primals_276, (24, ), (1, ))
    assert_size_stride(primals_280, (384, ), (1, ))
    assert_size_stride(primals_286, (384, ), (1, ))
    assert_size_stride(primals_292, (24, ), (1, ))
    assert_size_stride(primals_298, (24, ), (1, ))
    assert_size_stride(primals_304, (24, ), (1, ))
    assert_size_stride(primals_308, (384, ), (1, ))
    assert_size_stride(primals_314, (384, ), (1, ))
    assert_size_stride(primals_320, (24, ), (1, ))
    assert_size_stride(primals_326, (24, ), (1, ))
    assert_size_stride(primals_332, (24, ), (1, ))
    assert_size_stride(primals_336, (384, ), (1, ))
    assert_size_stride(primals_342, (384, ), (1, ))
    assert_size_stride(primals_348, (384, ), (1, ))
    assert_size_stride(primals_352, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(add, (4, 14), (14, 1))
    assert_size_stride(unsqueeze_5, (4, 14, 1, 1), (14, 1, 1, 1))
    assert_size_stride(clone_2, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(getitem_1, (8, 196, 1), (196, 1, 1))
    assert_size_stride(rsqrt, (8, 196, 1), (196, 1, 1))
    assert_size_stride(view_4, (1568, 384), (384, 1))
    assert_size_stride(addmm, (1568, 384), (384, 1))
    assert_size_stride(getitem_3, (8, 196, 1), (196, 1, 1))
    assert_size_stride(rsqrt_1, (8, 196, 1), (196, 1, 1))
    assert_size_stride(getitem_5, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(rsqrt_2, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(view_6, (25088, 24), (24, 1))
    assert_size_stride(view_19, (25088, 24), (24, 1))
    assert_size_stride(mul_7, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_21, (25088, 24), (24, 1))
    assert_size_stride(addmm_2, (25088, 96), (96, 1))
    assert_size_stride(view_23, (25088, 96), (96, 1))
    assert_size_stride(mul_12, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_26, (1568, 384), (384, 1))
    assert_size_stride(cat_1, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(getitem_13, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_5, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_28, (1576, 384), (384, 1))
    assert_size_stride(view_41, (1576, 384), (384, 1))
    assert_size_stride(mul_17, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(view_43, (1576, 384), (384, 1))
    assert_size_stride(addmm_6, (1576, 1536), (1536, 1))
    assert_size_stride(view_45, (1576, 1536), (1536, 1))
    assert_size_stride(view_47, (25088, 24), (24, 1))
    assert_size_stride(view_60, (25088, 24), (24, 1))
    assert_size_stride(mul_25, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_62, (25088, 24), (24, 1))
    assert_size_stride(addmm_9, (25088, 96), (96, 1))
    assert_size_stride(view_64, (25088, 96), (96, 1))
    assert_size_stride(mul_30, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_67, (1568, 384), (384, 1))
    assert_size_stride(cat_2, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(getitem_27, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_10, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_69, (1576, 384), (384, 1))
    assert_size_stride(view_82, (1576, 384), (384, 1))
    assert_size_stride(mul_35, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(view_84, (1576, 384), (384, 1))
    assert_size_stride(addmm_13, (1576, 1536), (1536, 1))
    assert_size_stride(view_86, (1576, 1536), (1536, 1))
    assert_size_stride(view_88, (25088, 24), (24, 1))
    assert_size_stride(view_101, (25088, 24), (24, 1))
    assert_size_stride(mul_43, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_103, (25088, 24), (24, 1))
    assert_size_stride(addmm_16, (25088, 96), (96, 1))
    assert_size_stride(view_105, (25088, 96), (96, 1))
    assert_size_stride(mul_48, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_108, (1568, 384), (384, 1))
    assert_size_stride(cat_3, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(getitem_41, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_15, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_110, (1576, 384), (384, 1))
    assert_size_stride(view_123, (1576, 384), (384, 1))
    assert_size_stride(mul_53, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(view_125, (1576, 384), (384, 1))
    assert_size_stride(addmm_20, (1576, 1536), (1536, 1))
    assert_size_stride(view_127, (1576, 1536), (1536, 1))
    assert_size_stride(view_129, (25088, 24), (24, 1))
    assert_size_stride(view_142, (25088, 24), (24, 1))
    assert_size_stride(mul_61, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_144, (25088, 24), (24, 1))
    assert_size_stride(addmm_23, (25088, 96), (96, 1))
    assert_size_stride(view_146, (25088, 96), (96, 1))
    assert_size_stride(mul_66, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_149, (1568, 384), (384, 1))
    assert_size_stride(cat_4, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(getitem_55, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_20, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_151, (1576, 384), (384, 1))
    assert_size_stride(view_164, (1576, 384), (384, 1))
    assert_size_stride(mul_71, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(view_166, (1576, 384), (384, 1))
    assert_size_stride(addmm_27, (1576, 1536), (1536, 1))
    assert_size_stride(view_168, (1576, 1536), (1536, 1))
    assert_size_stride(view_170, (25088, 24), (24, 1))
    assert_size_stride(view_183, (25088, 24), (24, 1))
    assert_size_stride(mul_79, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_185, (25088, 24), (24, 1))
    assert_size_stride(addmm_30, (25088, 96), (96, 1))
    assert_size_stride(view_187, (25088, 96), (96, 1))
    assert_size_stride(mul_84, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_190, (1568, 384), (384, 1))
    assert_size_stride(cat_5, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(getitem_69, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_25, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_192, (1576, 384), (384, 1))
    assert_size_stride(view_205, (1576, 384), (384, 1))
    assert_size_stride(mul_89, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(view_207, (1576, 384), (384, 1))
    assert_size_stride(addmm_34, (1576, 1536), (1536, 1))
    assert_size_stride(view_209, (1576, 1536), (1536, 1))
    assert_size_stride(view_211, (25088, 24), (24, 1))
    assert_size_stride(view_224, (25088, 24), (24, 1))
    assert_size_stride(mul_97, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_226, (25088, 24), (24, 1))
    assert_size_stride(addmm_37, (25088, 96), (96, 1))
    assert_size_stride(view_228, (25088, 96), (96, 1))
    assert_size_stride(mul_102, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_231, (1568, 384), (384, 1))
    assert_size_stride(cat_6, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(getitem_83, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_30, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_233, (1576, 384), (384, 1))
    assert_size_stride(view_246, (1576, 384), (384, 1))
    assert_size_stride(mul_107, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(view_248, (1576, 384), (384, 1))
    assert_size_stride(addmm_41, (1576, 1536), (1536, 1))
    assert_size_stride(view_250, (1576, 1536), (1536, 1))
    assert_size_stride(view_252, (25088, 24), (24, 1))
    assert_size_stride(view_265, (25088, 24), (24, 1))
    assert_size_stride(mul_115, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_267, (25088, 24), (24, 1))
    assert_size_stride(addmm_44, (25088, 96), (96, 1))
    assert_size_stride(view_269, (25088, 96), (96, 1))
    assert_size_stride(mul_120, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_272, (1568, 384), (384, 1))
    assert_size_stride(cat_7, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(getitem_97, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_35, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_274, (1576, 384), (384, 1))
    assert_size_stride(view_287, (1576, 384), (384, 1))
    assert_size_stride(mul_125, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(view_289, (1576, 384), (384, 1))
    assert_size_stride(addmm_48, (1576, 1536), (1536, 1))
    assert_size_stride(view_291, (1576, 1536), (1536, 1))
    assert_size_stride(view_293, (25088, 24), (24, 1))
    assert_size_stride(view_306, (25088, 24), (24, 1))
    assert_size_stride(mul_133, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_308, (25088, 24), (24, 1))
    assert_size_stride(addmm_51, (25088, 96), (96, 1))
    assert_size_stride(view_310, (25088, 96), (96, 1))
    assert_size_stride(mul_138, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_313, (1568, 384), (384, 1))
    assert_size_stride(cat_8, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(getitem_111, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_40, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_315, (1576, 384), (384, 1))
    assert_size_stride(view_328, (1576, 384), (384, 1))
    assert_size_stride(mul_143, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(view_330, (1576, 384), (384, 1))
    assert_size_stride(addmm_55, (1576, 1536), (1536, 1))
    assert_size_stride(view_332, (1576, 1536), (1536, 1))
    assert_size_stride(view_334, (25088, 24), (24, 1))
    assert_size_stride(view_347, (25088, 24), (24, 1))
    assert_size_stride(mul_151, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_349, (25088, 24), (24, 1))
    assert_size_stride(addmm_58, (25088, 96), (96, 1))
    assert_size_stride(view_351, (25088, 96), (96, 1))
    assert_size_stride(mul_156, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_354, (1568, 384), (384, 1))
    assert_size_stride(cat_9, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(getitem_125, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_45, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_356, (1576, 384), (384, 1))
    assert_size_stride(view_369, (1576, 384), (384, 1))
    assert_size_stride(mul_161, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(view_371, (1576, 384), (384, 1))
    assert_size_stride(addmm_62, (1576, 1536), (1536, 1))
    assert_size_stride(view_373, (1576, 1536), (1536, 1))
    assert_size_stride(view_375, (25088, 24), (24, 1))
    assert_size_stride(view_388, (25088, 24), (24, 1))
    assert_size_stride(mul_169, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_390, (25088, 24), (24, 1))
    assert_size_stride(addmm_65, (25088, 96), (96, 1))
    assert_size_stride(view_392, (25088, 96), (96, 1))
    assert_size_stride(mul_174, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_395, (1568, 384), (384, 1))
    assert_size_stride(cat_10, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(getitem_139, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_50, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_397, (1576, 384), (384, 1))
    assert_size_stride(view_410, (1576, 384), (384, 1))
    assert_size_stride(mul_179, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(view_412, (1576, 384), (384, 1))
    assert_size_stride(addmm_69, (1576, 1536), (1536, 1))
    assert_size_stride(view_414, (1576, 1536), (1536, 1))
    assert_size_stride(view_416, (25088, 24), (24, 1))
    assert_size_stride(view_429, (25088, 24), (24, 1))
    assert_size_stride(mul_187, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_431, (25088, 24), (24, 1))
    assert_size_stride(addmm_72, (25088, 96), (96, 1))
    assert_size_stride(view_433, (25088, 96), (96, 1))
    assert_size_stride(mul_192, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_436, (1568, 384), (384, 1))
    assert_size_stride(cat_11, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(getitem_153, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_55, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_438, (1576, 384), (384, 1))
    assert_size_stride(view_451, (1576, 384), (384, 1))
    assert_size_stride(mul_197, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(view_453, (1576, 384), (384, 1))
    assert_size_stride(addmm_76, (1576, 1536), (1536, 1))
    assert_size_stride(view_455, (1576, 1536), (1536, 1))
    assert_size_stride(view_457, (25088, 24), (24, 1))
    assert_size_stride(view_470, (25088, 24), (24, 1))
    assert_size_stride(mul_205, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_472, (25088, 24), (24, 1))
    assert_size_stride(addmm_79, (25088, 96), (96, 1))
    assert_size_stride(view_474, (25088, 96), (96, 1))
    assert_size_stride(mul_210, (1568, 16, 24), (384, 24, 1))
    assert_size_stride(view_477, (1568, 384), (384, 1))
    assert_size_stride(cat_12, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(getitem_167, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_60, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_479, (1576, 384), (384, 1))
    assert_size_stride(view_492, (1576, 384), (384, 1))
    assert_size_stride(mul_215, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(view_494, (1576, 384), (384, 1))
    assert_size_stride(addmm_83, (1576, 1536), (1536, 1))
    assert_size_stride(view_496, (1576, 1536), (1536, 1))
    assert_size_stride(mul_220, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(clone_184, (8, 384), (384, 1))
    assert_size_stride(permute_233, (1000, 384), (384, 1))
    assert_size_stride(div_24, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_237, (384, 1536), (1536, 1))
    assert_size_stride(permute_241, (1536, 384), (384, 1))
    assert_size_stride(div_25, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_245, (384, 384), (384, 1))
    assert_size_stride(permute_250, (48, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_251, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_24, (8, 6, 197, 197), (232854, 38809, 197, 1))
    assert_size_stride(permute_252, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_253, (48, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_258, (384, 384), (384, 1))
    assert_size_stride(permute_263, (768, 384), (384, 1))
    assert_size_stride(permute_265, (384, 384), (384, 1))
    assert_size_stride(div_27, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_269, (24, 96), (96, 1))
    assert_size_stride(permute_273, (96, 24), (24, 1))
    assert_size_stride(div_28, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_277, (24, 24), (24, 1))
    assert_size_stride(permute_282, (6272, 16, 16), (256, 1, 16))
    assert_size_stride(permute_283, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(alias_25, (1568, 4, 16, 16), (1024, 256, 16, 1))
    assert_size_stride(permute_284, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(permute_285, (6272, 16, 6), (96, 1, 16))
    assert_size_stride(permute_290, (24, 24), (24, 1))
    assert_size_stride(permute_295, (48, 24), (24, 1))
    assert_size_stride(div_29, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_297, (384, 1536), (1536, 1))
    assert_size_stride(permute_301, (1536, 384), (384, 1))
    assert_size_stride(div_30, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_305, (384, 384), (384, 1))
    assert_size_stride(permute_310, (48, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_311, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_26, (8, 6, 197, 197), (232854, 38809, 197, 1))
    assert_size_stride(permute_312, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_313, (48, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_318, (384, 384), (384, 1))
    assert_size_stride(permute_323, (768, 384), (384, 1))
    assert_size_stride(permute_325, (384, 384), (384, 1))
    assert_size_stride(permute_329, (24, 96), (96, 1))
    assert_size_stride(permute_333, (96, 24), (24, 1))
    assert_size_stride(div_33, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_337, (24, 24), (24, 1))
    assert_size_stride(permute_342, (6272, 16, 16), (256, 1, 16))
    assert_size_stride(permute_343, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(alias_27, (1568, 4, 16, 16), (1024, 256, 16, 1))
    assert_size_stride(permute_344, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(permute_345, (6272, 16, 6), (96, 1, 16))
    assert_size_stride(permute_350, (24, 24), (24, 1))
    assert_size_stride(permute_355, (48, 24), (24, 1))
    assert_size_stride(div_34, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_357, (384, 1536), (1536, 1))
    assert_size_stride(permute_361, (1536, 384), (384, 1))
    assert_size_stride(div_35, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_365, (384, 384), (384, 1))
    assert_size_stride(permute_370, (48, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_371, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_28, (8, 6, 197, 197), (232854, 38809, 197, 1))
    assert_size_stride(permute_372, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_373, (48, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_378, (384, 384), (384, 1))
    assert_size_stride(permute_383, (768, 384), (384, 1))
    assert_size_stride(permute_385, (384, 384), (384, 1))
    assert_size_stride(permute_389, (24, 96), (96, 1))
    assert_size_stride(permute_393, (96, 24), (24, 1))
    assert_size_stride(div_38, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_397, (24, 24), (24, 1))
    assert_size_stride(permute_402, (6272, 16, 16), (256, 1, 16))
    assert_size_stride(permute_403, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(alias_29, (1568, 4, 16, 16), (1024, 256, 16, 1))
    assert_size_stride(permute_404, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(permute_405, (6272, 16, 6), (96, 1, 16))
    assert_size_stride(permute_410, (24, 24), (24, 1))
    assert_size_stride(permute_415, (48, 24), (24, 1))
    assert_size_stride(div_39, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_417, (384, 1536), (1536, 1))
    assert_size_stride(permute_421, (1536, 384), (384, 1))
    assert_size_stride(div_40, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_425, (384, 384), (384, 1))
    assert_size_stride(permute_430, (48, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_431, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_30, (8, 6, 197, 197), (232854, 38809, 197, 1))
    assert_size_stride(permute_432, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_433, (48, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_438, (384, 384), (384, 1))
    assert_size_stride(permute_443, (768, 384), (384, 1))
    assert_size_stride(permute_445, (384, 384), (384, 1))
    assert_size_stride(permute_449, (24, 96), (96, 1))
    assert_size_stride(permute_453, (96, 24), (24, 1))
    assert_size_stride(div_43, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_457, (24, 24), (24, 1))
    assert_size_stride(permute_462, (6272, 16, 16), (256, 1, 16))
    assert_size_stride(permute_463, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(alias_31, (1568, 4, 16, 16), (1024, 256, 16, 1))
    assert_size_stride(permute_464, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(permute_465, (6272, 16, 6), (96, 1, 16))
    assert_size_stride(permute_470, (24, 24), (24, 1))
    assert_size_stride(permute_475, (48, 24), (24, 1))
    assert_size_stride(div_44, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_477, (384, 1536), (1536, 1))
    assert_size_stride(permute_481, (1536, 384), (384, 1))
    assert_size_stride(div_45, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_485, (384, 384), (384, 1))
    assert_size_stride(permute_490, (48, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_491, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_32, (8, 6, 197, 197), (232854, 38809, 197, 1))
    assert_size_stride(permute_492, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_493, (48, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_498, (384, 384), (384, 1))
    assert_size_stride(permute_503, (768, 384), (384, 1))
    assert_size_stride(permute_505, (384, 384), (384, 1))
    assert_size_stride(permute_509, (24, 96), (96, 1))
    assert_size_stride(permute_513, (96, 24), (24, 1))
    assert_size_stride(div_48, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_517, (24, 24), (24, 1))
    assert_size_stride(permute_522, (6272, 16, 16), (256, 1, 16))
    assert_size_stride(permute_523, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(alias_33, (1568, 4, 16, 16), (1024, 256, 16, 1))
    assert_size_stride(permute_524, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(permute_525, (6272, 16, 6), (96, 1, 16))
    assert_size_stride(permute_530, (24, 24), (24, 1))
    assert_size_stride(permute_535, (48, 24), (24, 1))
    assert_size_stride(div_49, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_537, (384, 1536), (1536, 1))
    assert_size_stride(permute_541, (1536, 384), (384, 1))
    assert_size_stride(div_50, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_545, (384, 384), (384, 1))
    assert_size_stride(permute_550, (48, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_551, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_34, (8, 6, 197, 197), (232854, 38809, 197, 1))
    assert_size_stride(permute_552, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_553, (48, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_558, (384, 384), (384, 1))
    assert_size_stride(permute_563, (768, 384), (384, 1))
    assert_size_stride(permute_565, (384, 384), (384, 1))
    assert_size_stride(permute_569, (24, 96), (96, 1))
    assert_size_stride(permute_573, (96, 24), (24, 1))
    assert_size_stride(div_53, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_577, (24, 24), (24, 1))
    assert_size_stride(permute_582, (6272, 16, 16), (256, 1, 16))
    assert_size_stride(permute_583, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(alias_35, (1568, 4, 16, 16), (1024, 256, 16, 1))
    assert_size_stride(permute_584, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(permute_585, (6272, 16, 6), (96, 1, 16))
    assert_size_stride(permute_590, (24, 24), (24, 1))
    assert_size_stride(permute_595, (48, 24), (24, 1))
    assert_size_stride(div_54, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_597, (384, 1536), (1536, 1))
    assert_size_stride(permute_601, (1536, 384), (384, 1))
    assert_size_stride(div_55, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_605, (384, 384), (384, 1))
    assert_size_stride(permute_610, (48, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_611, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_36, (8, 6, 197, 197), (232854, 38809, 197, 1))
    assert_size_stride(permute_612, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_613, (48, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_618, (384, 384), (384, 1))
    assert_size_stride(permute_623, (768, 384), (384, 1))
    assert_size_stride(permute_625, (384, 384), (384, 1))
    assert_size_stride(permute_629, (24, 96), (96, 1))
    assert_size_stride(permute_633, (96, 24), (24, 1))
    assert_size_stride(div_58, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_637, (24, 24), (24, 1))
    assert_size_stride(permute_642, (6272, 16, 16), (256, 1, 16))
    assert_size_stride(permute_643, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(alias_37, (1568, 4, 16, 16), (1024, 256, 16, 1))
    assert_size_stride(permute_644, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(permute_645, (6272, 16, 6), (96, 1, 16))
    assert_size_stride(permute_650, (24, 24), (24, 1))
    assert_size_stride(permute_655, (48, 24), (24, 1))
    assert_size_stride(div_59, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_657, (384, 1536), (1536, 1))
    assert_size_stride(permute_661, (1536, 384), (384, 1))
    assert_size_stride(div_60, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_665, (384, 384), (384, 1))
    assert_size_stride(permute_670, (48, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_671, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_38, (8, 6, 197, 197), (232854, 38809, 197, 1))
    assert_size_stride(permute_672, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_673, (48, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_678, (384, 384), (384, 1))
    assert_size_stride(permute_683, (768, 384), (384, 1))
    assert_size_stride(permute_685, (384, 384), (384, 1))
    assert_size_stride(permute_689, (24, 96), (96, 1))
    assert_size_stride(permute_693, (96, 24), (24, 1))
    assert_size_stride(div_63, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_697, (24, 24), (24, 1))
    assert_size_stride(permute_702, (6272, 16, 16), (256, 1, 16))
    assert_size_stride(permute_703, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(alias_39, (1568, 4, 16, 16), (1024, 256, 16, 1))
    assert_size_stride(permute_704, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(permute_705, (6272, 16, 6), (96, 1, 16))
    assert_size_stride(permute_710, (24, 24), (24, 1))
    assert_size_stride(permute_715, (48, 24), (24, 1))
    assert_size_stride(div_64, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_717, (384, 1536), (1536, 1))
    assert_size_stride(permute_721, (1536, 384), (384, 1))
    assert_size_stride(div_65, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_725, (384, 384), (384, 1))
    assert_size_stride(permute_730, (48, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_731, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_40, (8, 6, 197, 197), (232854, 38809, 197, 1))
    assert_size_stride(permute_732, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_733, (48, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_738, (384, 384), (384, 1))
    assert_size_stride(permute_743, (768, 384), (384, 1))
    assert_size_stride(permute_745, (384, 384), (384, 1))
    assert_size_stride(permute_749, (24, 96), (96, 1))
    assert_size_stride(permute_753, (96, 24), (24, 1))
    assert_size_stride(div_68, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_757, (24, 24), (24, 1))
    assert_size_stride(permute_762, (6272, 16, 16), (256, 1, 16))
    assert_size_stride(permute_763, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(alias_41, (1568, 4, 16, 16), (1024, 256, 16, 1))
    assert_size_stride(permute_764, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(permute_765, (6272, 16, 6), (96, 1, 16))
    assert_size_stride(permute_770, (24, 24), (24, 1))
    assert_size_stride(permute_775, (48, 24), (24, 1))
    assert_size_stride(div_69, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_777, (384, 1536), (1536, 1))
    assert_size_stride(permute_781, (1536, 384), (384, 1))
    assert_size_stride(div_70, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_785, (384, 384), (384, 1))
    assert_size_stride(permute_790, (48, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_791, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_42, (8, 6, 197, 197), (232854, 38809, 197, 1))
    assert_size_stride(permute_792, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_793, (48, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_798, (384, 384), (384, 1))
    assert_size_stride(permute_803, (768, 384), (384, 1))
    assert_size_stride(permute_805, (384, 384), (384, 1))
    assert_size_stride(permute_809, (24, 96), (96, 1))
    assert_size_stride(permute_813, (96, 24), (24, 1))
    assert_size_stride(div_73, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_817, (24, 24), (24, 1))
    assert_size_stride(permute_822, (6272, 16, 16), (256, 1, 16))
    assert_size_stride(permute_823, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(alias_43, (1568, 4, 16, 16), (1024, 256, 16, 1))
    assert_size_stride(permute_824, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(permute_825, (6272, 16, 6), (96, 1, 16))
    assert_size_stride(permute_830, (24, 24), (24, 1))
    assert_size_stride(permute_835, (48, 24), (24, 1))
    assert_size_stride(div_74, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_837, (384, 1536), (1536, 1))
    assert_size_stride(permute_841, (1536, 384), (384, 1))
    assert_size_stride(div_75, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_845, (384, 384), (384, 1))
    assert_size_stride(permute_850, (48, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_851, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_44, (8, 6, 197, 197), (232854, 38809, 197, 1))
    assert_size_stride(permute_852, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_853, (48, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_858, (384, 384), (384, 1))
    assert_size_stride(permute_863, (768, 384), (384, 1))
    assert_size_stride(permute_865, (384, 384), (384, 1))
    assert_size_stride(permute_869, (24, 96), (96, 1))
    assert_size_stride(permute_873, (96, 24), (24, 1))
    assert_size_stride(div_78, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_877, (24, 24), (24, 1))
    assert_size_stride(permute_882, (6272, 16, 16), (256, 1, 16))
    assert_size_stride(permute_883, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(alias_45, (1568, 4, 16, 16), (1024, 256, 16, 1))
    assert_size_stride(permute_884, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(permute_885, (6272, 16, 6), (96, 1, 16))
    assert_size_stride(permute_890, (24, 24), (24, 1))
    assert_size_stride(permute_895, (48, 24), (24, 1))
    assert_size_stride(div_79, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_897, (384, 1536), (1536, 1))
    assert_size_stride(permute_901, (1536, 384), (384, 1))
    assert_size_stride(div_80, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_905, (384, 384), (384, 1))
    assert_size_stride(permute_910, (48, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_911, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_46, (8, 6, 197, 197), (232854, 38809, 197, 1))
    assert_size_stride(permute_912, (48, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_913, (48, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_918, (384, 384), (384, 1))
    assert_size_stride(permute_923, (768, 384), (384, 1))
    assert_size_stride(permute_925, (384, 384), (384, 1))
    assert_size_stride(permute_929, (24, 96), (96, 1))
    assert_size_stride(permute_933, (96, 24), (24, 1))
    assert_size_stride(div_83, (1568, 16, 1), (16, 1, 1))
    assert_size_stride(permute_937, (24, 24), (24, 1))
    assert_size_stride(permute_942, (6272, 16, 16), (256, 1, 16))
    assert_size_stride(permute_943, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(alias_47, (1568, 4, 16, 16), (1024, 256, 16, 1))
    assert_size_stride(permute_944, (6272, 6, 16), (96, 1, 6))
    assert_size_stride(permute_945, (6272, 16, 6), (96, 1, 16))
    assert_size_stride(permute_950, (24, 24), (24, 1))
    assert_size_stride(permute_955, (48, 24), (24, 1))
    assert_size_stride(permute_957, (384, 384), (384, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_233, out=buf0)
        del permute_233
        buf1 = empty((1000, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_184, out=buf1)
        del clone_184
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf5 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_backward_select_backward_1.run(buf0, primals_348, mul_220, div_24, buf5, 1576, 384, grid=grid(1576), stream=stream0)
        del div_24
        del primals_348
        buf6 = empty_strided((384, 13), (1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_backward_select_backward_2.run(buf0, mul_220, buf6, 4992, 122, grid=grid(4992), stream=stream0)
        del mul_220
        buf7 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf6, buf7, 384, 13, grid=grid(384), stream=stream0)
        buf8 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_backward_select_backward_4.run(buf0, buf8, 384, 1576, grid=grid(384), stream=stream0)
        del buf0
        buf9 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1576, 384), (384, 1), 0), permute_237, out=buf9)
        del permute_237
        buf10 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (384, 1576), (1, 384), 0), view_496, out=buf10)
        del view_496
        buf11 = reinterpret_tensor(buf6, (1, 384, 13), (4992, 1, 384), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf5, buf11, 4992, 122, grid=grid(4992), stream=stream0)
        buf12 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf11, buf12, 384, 13, grid=grid(384), stream=stream0)
        buf13 = reinterpret_tensor(buf9, (8, 197, 1536), (302592, 1536, 1), 0); del buf9  # reuse
        # Source Nodes: [x_216], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf13, addmm_83, 2420736, grid=grid(2420736), stream=stream0)
        del addmm_83
        buf14 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (1576, 1536), (1536, 1), 0), permute_241, out=buf14)
        del permute_241
        buf15 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (1536, 1576), (1, 1536), 0), view_494, out=buf15)
        del view_494
        buf16 = empty_strided((1, 1536, 13), (19968, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf13, buf16, 19968, 122, grid=grid(19968), stream=stream0)
        buf17 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf16, buf17, 1536, 13, grid=grid(1536), stream=stream0)
        buf24 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf24, buf14, primals_342, mul_215, div_25, 1576, 384, grid=grid(1576), stream=stream0)
        del div_25
        del primals_342
        buf20 = reinterpret_tensor(buf11, (384, 13), (1, 384), 0); del buf11  # reuse
        buf22 = empty_strided((384, 13), (1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf14, mul_215, buf20, buf22, 4992, 122, grid=grid(4992), stream=stream0)
        del mul_215
        buf21 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf20, buf21, 384, 13, grid=grid(384), stream=stream0)
        buf23 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf22, buf23, 384, 13, grid=grid(384), stream=stream0)
        buf25 = buf14; del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (1576, 384), (384, 1), 0), permute_245, out=buf25)
        del permute_245
        buf26 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (384, 1576), (1, 384), 0), view_492, out=buf26)
        del view_492
        buf27 = reinterpret_tensor(buf22, (1, 384, 13), (4992, 1, 384), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf24, buf27, 4992, 122, grid=grid(4992), stream=stream0)
        buf28 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf27, buf28, 384, 13, grid=grid(384), stream=stream0)
        buf29 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf25, buf29, 605184, grid=grid(605184), stream=stream0)
        buf30 = reinterpret_tensor(buf25, (48, 197, 64), (12608, 64, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_250, reinterpret_tensor(buf29, (48, 197, 64), (12608, 64, 1), 0), out=buf30)
        del permute_250
        buf31 = empty((48, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf29, (48, 197, 64), (12608, 64, 1), 0), permute_251, out=buf31)
        del permute_251
        buf33 = empty((8, 6, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_12.run(buf31, alias_24, buf33, 9456, 197, grid=grid(9456), stream=stream0)
        del alias_24
        buf34 = reinterpret_tensor(buf29, (48, 64, 197), (12608, 197, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_252, reinterpret_tensor(buf33, (48, 197, 197), (38809, 197, 1), 0), out=buf34)
        del permute_252
        buf35 = empty((48, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf33, (48, 197, 197), (38809, 197, 1), 0), permute_253, out=buf35)
        del permute_253
        buf36 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf30, buf36, 605184, grid=grid(605184), stream=stream0)
        buf37 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (384, 1576), (1, 384), 0), view_479, out=buf37)
        buf38 = reinterpret_tensor(buf30, (1576, 384), (384, 1), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf36, permute_258, out=buf38)
        del permute_258
        buf39 = empty((8, 197, 2, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf35, buf34, buf39, 18912, 64, grid=grid(18912, 64), stream=stream0)
        buf40 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (768, 1576), (1, 768), 0), view_479, out=buf40)
        del view_479
        buf41 = reinterpret_tensor(buf35, (1576, 384), (384, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (1576, 768), (768, 1), 0), permute_263, out=buf41)
        del permute_263
        buf48 = buf24; del buf24  # reuse
        # Source Nodes: [l__mod___blocks_11_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_15.run(buf48, buf38, buf41, primals_336, cat_12, getitem_167, rsqrt_60, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_336
        buf44 = reinterpret_tensor(buf27, (384, 13), (1, 384), 0); del buf27  # reuse
        buf46 = buf20; del buf20  # reuse
        # Source Nodes: [l__mod___blocks_11_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf38, buf41, cat_12, getitem_167, rsqrt_60, buf44, buf46, 4992, 122, grid=grid(4992), stream=stream0)
        del cat_12
        del getitem_167
        del rsqrt_60
        buf45 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_11_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf44, buf45, 384, 13, grid=grid(384), stream=stream0)
        buf47 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf46, buf47, 384, 13, grid=grid(384), stream=stream0)
        buf49 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_17.run(buf48, buf49, 602112, grid=grid(602112), stream=stream0)
        buf50 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf49, permute_265, out=buf50)
        del permute_265
        buf51 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (384, 1568), (1, 384), 0), view_477, out=buf51)
        del view_477
        buf52 = reinterpret_tensor(buf46, (1, 384, 13), (4992, 1, 384), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf49, buf52, 4992, 121, grid=grid(4992), stream=stream0)
        buf53 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf52, buf53, 384, 13, grid=grid(384), stream=stream0)
        buf56 = reinterpret_tensor(buf49, (1568, 16, 24), (384, 24, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_19.run(buf50, primals_332, mul_210, div_27, buf56, 25088, 24, grid=grid(25088), stream=stream0)
        del div_27
        del primals_332
        buf57 = empty_strided((24, 196), (1, 24), device='cuda', dtype=torch.float32)
        buf59 = empty_strided((24, 196), (1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_20.run(buf50, mul_210, buf57, buf59, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_210
        buf58 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf57, buf58, 24, 196, grid=grid(24), stream=stream0)
        buf60 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf59, buf60, 24, 196, grid=grid(24), stream=stream0)
        buf61 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf56, (25088, 24), (24, 1), 0), permute_269, out=buf61)
        del permute_269
        buf62 = empty((24, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf56, (24, 25088), (1, 24), 0), view_474, out=buf62)
        del view_474
        buf63 = reinterpret_tensor(buf59, (1, 24, 196), (4704, 1, 24), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf56, buf63, 4704, 128, grid=grid(4704), stream=stream0)
        buf64 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf63, buf64, 24, 196, grid=grid(24), stream=stream0)
        buf65 = reinterpret_tensor(buf61, (1568, 16, 96), (1536, 96, 1), 0); del buf61  # reuse
        # Source Nodes: [x_207], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf65, addmm_79, 2408448, grid=grid(2408448), stream=stream0)
        del addmm_79
        buf66 = reinterpret_tensor(buf50, (25088, 24), (24, 1), 0); del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (25088, 96), (96, 1), 0), permute_273, out=buf66)
        del permute_273
        buf67 = empty((96, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (96, 25088), (1, 96), 0), view_472, out=buf67)
        del view_472
        buf68 = empty_strided((1, 96, 196), (18816, 1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf65, buf68, 18816, 128, grid=grid(18816), stream=stream0)
        buf69 = empty((1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf68, buf69, 96, 196, grid=grid(96), stream=stream0)
        buf76 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_26.run(buf76, buf66, primals_326, mul_205, div_28, 25088, 24, grid=grid(25088), stream=stream0)
        del div_28
        del primals_326
        buf72 = reinterpret_tensor(buf63, (24, 196), (1, 24), 0); del buf63  # reuse
        buf74 = buf57; del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_20.run(buf66, mul_205, buf72, buf74, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_205
        buf73 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf72, buf73, 24, 196, grid=grid(24), stream=stream0)
        buf75 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf74, buf75, 24, 196, grid=grid(24), stream=stream0)
        buf77 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (25088, 24), (24, 1), 0), permute_277, out=buf77)
        del permute_277
        buf78 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (24, 25088), (1, 24), 0), view_470, out=buf78)
        del view_470
        buf79 = reinterpret_tensor(buf74, (1, 24, 196), (4704, 1, 24), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf76, buf79, 4704, 128, grid=grid(4704), stream=stream0)
        buf80 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf79, buf80, 24, 196, grid=grid(24), stream=stream0)
        buf81 = empty((1568, 4, 16, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf77, buf81, 602112, grid=grid(602112), stream=stream0)
        buf82 = reinterpret_tensor(buf77, (6272, 16, 6), (96, 6, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_282, reinterpret_tensor(buf81, (6272, 16, 6), (96, 6, 1), 0), out=buf82)
        del permute_282
        buf83 = empty((6272, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf81, (6272, 16, 6), (96, 6, 1), 0), permute_283, out=buf83)
        del permute_283
        buf85 = empty((1568, 4, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_28.run(buf83, alias_25, buf85, 100352, 16, grid=grid(100352), stream=stream0)
        del alias_25
        buf86 = reinterpret_tensor(buf81, (6272, 6, 16), (96, 16, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_284, reinterpret_tensor(buf85, (6272, 16, 16), (256, 16, 1), 0), out=buf86)
        del permute_284
        buf87 = empty((6272, 16, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf85, (6272, 16, 16), (256, 16, 1), 0), permute_285, out=buf87)
        del permute_285
        buf88 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_29.run(buf82, buf88, 602112, grid=grid(602112), stream=stream0)
        buf89 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (24, 25088), (1, 24), 0), view_457, out=buf89)
        buf90 = reinterpret_tensor(buf82, (25088, 24), (24, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf88, permute_290, out=buf90)
        del permute_290
        buf91 = empty((1568, 16, 2, 4, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf87, buf86, buf91, 200704, 6, grid=grid(200704, 6), stream=stream0)
        buf92 = empty((48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (48, 25088), (1, 48), 0), view_457, out=buf92)
        del view_457
        buf93 = reinterpret_tensor(buf87, (25088, 24), (24, 1), 0); del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (25088, 48), (48, 1), 0), permute_295, out=buf93)
        del permute_295
        buf100 = reinterpret_tensor(buf41, (8, 197, 384), (75648, 384, 1), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_select_backward_slice_backward_31.run(buf48, buf100, 605184, grid=grid(605184), stream=stream0)
        buf101 = reinterpret_tensor(buf13, (1576, 1536), (1536, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (1576, 384), (384, 1), 0), permute_297, out=buf101)
        del permute_297
        buf105 = reinterpret_tensor(buf101, (8, 197, 1536), (302592, 1536, 1), 0); del buf101  # reuse
        # Source Nodes: [x_198], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf105, addmm_76, 2420736, grid=grid(2420736), stream=stream0)
        del addmm_76
        buf106 = reinterpret_tensor(buf48, (1576, 384), (384, 1), 0); del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (1576, 1536), (1536, 1), 0), permute_301, out=buf106)
        del permute_301
        buf116 = reinterpret_tensor(buf38, (8, 197, 384), (75648, 384, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf106, primals_314, mul_197, buf100, div_30, buf116, 1576, 384, grid=grid(1576), stream=stream0)
        del div_30
        del primals_314
        buf117 = reinterpret_tensor(buf34, (1576, 384), (384, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (1576, 384), (384, 1), 0), permute_305, out=buf117)
        del permute_305
        buf121 = reinterpret_tensor(buf36, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf117, buf121, 605184, grid=grid(605184), stream=stream0)
        buf122 = reinterpret_tensor(buf117, (48, 197, 64), (12608, 64, 1), 0); del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_310, reinterpret_tensor(buf121, (48, 197, 64), (12608, 64, 1), 0), out=buf122)
        del permute_310
        buf128 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf122, buf128, 605184, grid=grid(605184), stream=stream0)
        buf130 = reinterpret_tensor(buf122, (1576, 384), (384, 1), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf128, permute_318, out=buf130)
        del permute_318
        buf123 = reinterpret_tensor(buf33, (48, 197, 197), (38809, 197, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf121, (48, 197, 64), (12608, 64, 1), 0), permute_311, out=buf123)
        del permute_311
        buf125 = reinterpret_tensor(buf31, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_12.run(buf123, alias_26, buf125, 9456, 197, grid=grid(9456), stream=stream0)
        del alias_26
        buf126 = reinterpret_tensor(buf121, (48, 64, 197), (12608, 197, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_312, reinterpret_tensor(buf125, (48, 197, 197), (38809, 197, 1), 0), out=buf126)
        del permute_312
        buf127 = empty((48, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf125, (48, 197, 197), (38809, 197, 1), 0), permute_313, out=buf127)
        del permute_313
        buf131 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf127, buf126, buf131, 18912, 64, grid=grid(18912, 64), stream=stream0)
        buf133 = reinterpret_tensor(buf127, (1576, 384), (384, 1), 0); del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf131, (1576, 768), (768, 1), 0), permute_323, out=buf133)
        del permute_323
        buf140 = reinterpret_tensor(buf126, (8, 197, 384), (75648, 384, 1), 0); del buf126  # reuse
        # Source Nodes: [l__mod___blocks_10_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33.run(buf130, buf133, primals_308, cat_11, getitem_153, rsqrt_55, buf116, buf140, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_308
        buf141 = reinterpret_tensor(buf86, (1568, 384), (384, 1), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_17.run(buf140, buf141, 602112, grid=grid(602112), stream=stream0)
        buf142 = reinterpret_tensor(buf88, (1568, 384), (384, 1), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf141, permute_325, out=buf142)
        del permute_325
        buf152 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_34.run(buf152, buf90, buf93, primals_320, mul_192, buf142, primals_304, div_29, 25088, 24, grid=grid(25088), stream=stream0)
        del div_29
        del primals_304
        del primals_320
        buf96 = reinterpret_tensor(buf79, (24, 196), (1, 24), 0); del buf79  # reuse
        buf98 = buf72; del buf72  # reuse
        buf148 = empty_strided((24, 196), (1, 24), device='cuda', dtype=torch.float32)
        buf150 = empty_strided((24, 196), (1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_35.run(buf90, buf93, mul_192, buf142, buf96, buf98, buf148, buf150, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_192
        buf97 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf96, buf97, 24, 196, grid=grid(24), stream=stream0)
        buf99 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf98, buf99, 24, 196, grid=grid(24), stream=stream0)
        buf102 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (384, 1576), (1, 384), 0), view_455, out=buf102)
        del view_455
        buf103 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf100, buf103, 4992, 122, grid=grid(4992), stream=stream0)
        buf104 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf103, buf104, 384, 13, grid=grid(384), stream=stream0)
        buf107 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (1536, 1576), (1, 1536), 0), view_453, out=buf107)
        del view_453
        buf108 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf105, buf108, 19968, 122, grid=grid(19968), stream=stream0)
        buf109 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf108, buf109, 1536, 13, grid=grid(1536), stream=stream0)
        buf112 = reinterpret_tensor(buf103, (384, 13), (1, 384), 0); del buf103  # reuse
        buf114 = buf44; del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf106, mul_197, buf112, buf114, 4992, 122, grid=grid(4992), stream=stream0)
        del mul_197
        buf113 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf112, buf113, 384, 13, grid=grid(384), stream=stream0)
        buf115 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf114, buf115, 384, 13, grid=grid(384), stream=stream0)
        buf118 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (384, 1576), (1, 384), 0), view_451, out=buf118)
        del view_451
        buf119 = reinterpret_tensor(buf114, (1, 384, 13), (4992, 1, 384), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf116, buf119, 4992, 122, grid=grid(4992), stream=stream0)
        buf120 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf119, buf120, 384, 13, grid=grid(384), stream=stream0)
        buf129 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (384, 1576), (1, 384), 0), view_438, out=buf129)
        buf132 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf131, (768, 1576), (1, 768), 0), view_438, out=buf132)
        del view_438
        buf136 = reinterpret_tensor(buf119, (384, 13), (1, 384), 0); del buf119  # reuse
        buf138 = buf112; del buf112  # reuse
        # Source Nodes: [l__mod___blocks_10_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf130, buf133, cat_11, getitem_153, rsqrt_55, buf136, buf138, 4992, 122, grid=grid(4992), stream=stream0)
        del cat_11
        del getitem_153
        del rsqrt_55
        buf137 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_10_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf136, buf137, 384, 13, grid=grid(384), stream=stream0)
        buf139 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf138, buf139, 384, 13, grid=grid(384), stream=stream0)
        buf143 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (384, 1568), (1, 384), 0), view_436, out=buf143)
        del view_436
        buf144 = reinterpret_tensor(buf138, (1, 384, 13), (4992, 1, 384), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf141, buf144, 4992, 121, grid=grid(4992), stream=stream0)
        buf145 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf144, buf145, 384, 13, grid=grid(384), stream=stream0)
        buf149 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf148, buf149, 24, 196, grid=grid(24), stream=stream0)
        buf151 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf150, buf151, 24, 196, grid=grid(24), stream=stream0)
        buf153 = reinterpret_tensor(buf65, (25088, 96), (96, 1), 0); del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (25088, 24), (24, 1), 0), permute_329, out=buf153)
        del permute_329
        buf154 = empty((24, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (24, 25088), (1, 24), 0), view_433, out=buf154)
        del view_433
        buf155 = reinterpret_tensor(buf150, (1, 24, 196), (4704, 1, 24), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf152, buf155, 4704, 128, grid=grid(4704), stream=stream0)
        buf156 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf155, buf156, 24, 196, grid=grid(24), stream=stream0)
        buf157 = reinterpret_tensor(buf153, (1568, 16, 96), (1536, 96, 1), 0); del buf153  # reuse
        # Source Nodes: [x_189], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf157, addmm_72, 2408448, grid=grid(2408448), stream=stream0)
        del addmm_72
        buf158 = reinterpret_tensor(buf141, (25088, 24), (24, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (25088, 96), (96, 1), 0), permute_333, out=buf158)
        del permute_333
        buf159 = empty((96, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (96, 25088), (1, 96), 0), view_431, out=buf159)
        del view_431
        buf160 = buf68; del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf157, buf160, 18816, 128, grid=grid(18816), stream=stream0)
        buf161 = empty((1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf160, buf161, 96, 196, grid=grid(96), stream=stream0)
        buf168 = buf152; del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_26.run(buf168, buf158, primals_298, mul_187, div_33, 25088, 24, grid=grid(25088), stream=stream0)
        del div_33
        del primals_298
        buf164 = reinterpret_tensor(buf155, (24, 196), (1, 24), 0); del buf155  # reuse
        buf166 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_20.run(buf158, mul_187, buf164, buf166, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_187
        buf165 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf164, buf165, 24, 196, grid=grid(24), stream=stream0)
        buf167 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf166, buf167, 24, 196, grid=grid(24), stream=stream0)
        buf169 = buf158; del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf168, (25088, 24), (24, 1), 0), permute_337, out=buf169)
        del permute_337
        buf170 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf168, (24, 25088), (1, 24), 0), view_429, out=buf170)
        del view_429
        buf171 = reinterpret_tensor(buf166, (1, 24, 196), (4704, 1, 24), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf168, buf171, 4704, 128, grid=grid(4704), stream=stream0)
        buf172 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf171, buf172, 24, 196, grid=grid(24), stream=stream0)
        buf173 = reinterpret_tensor(buf93, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf169, buf173, 602112, grid=grid(602112), stream=stream0)
        buf174 = reinterpret_tensor(buf169, (6272, 16, 6), (96, 6, 1), 0); del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_342, reinterpret_tensor(buf173, (6272, 16, 6), (96, 6, 1), 0), out=buf174)
        del permute_342
        buf175 = reinterpret_tensor(buf85, (6272, 16, 16), (256, 16, 1), 0); del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf173, (6272, 16, 6), (96, 6, 1), 0), permute_343, out=buf175)
        del permute_343
        buf177 = reinterpret_tensor(buf83, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_28.run(buf175, alias_27, buf177, 100352, 16, grid=grid(100352), stream=stream0)
        del alias_27
        buf178 = reinterpret_tensor(buf173, (6272, 6, 16), (96, 16, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_344, reinterpret_tensor(buf177, (6272, 16, 16), (256, 16, 1), 0), out=buf178)
        del permute_344
        buf179 = reinterpret_tensor(buf90, (6272, 16, 6), (96, 6, 1), 0); del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf177, (6272, 16, 16), (256, 16, 1), 0), permute_345, out=buf179)
        del permute_345
        buf180 = reinterpret_tensor(buf142, (25088, 24), (24, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_29.run(buf174, buf180, 602112, grid=grid(602112), stream=stream0)
        buf181 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf180, (24, 25088), (1, 24), 0), view_416, out=buf181)
        buf182 = reinterpret_tensor(buf174, (25088, 24), (24, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf180, permute_350, out=buf182)
        del permute_350
        buf183 = buf91; del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf179, buf178, buf183, 200704, 6, grid=grid(200704, 6), stream=stream0)
        buf184 = empty((48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (48, 25088), (1, 48), 0), view_416, out=buf184)
        del view_416
        buf185 = reinterpret_tensor(buf179, (25088, 24), (24, 1), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (25088, 48), (48, 1), 0), permute_355, out=buf185)
        del permute_355
        buf192 = reinterpret_tensor(buf133, (8, 197, 384), (75648, 384, 1), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_select_backward_slice_backward_31.run(buf140, buf192, 605184, grid=grid(605184), stream=stream0)
        buf193 = reinterpret_tensor(buf105, (1576, 1536), (1536, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf192, (1576, 384), (384, 1), 0), permute_357, out=buf193)
        del permute_357
        buf197 = reinterpret_tensor(buf193, (8, 197, 1536), (302592, 1536, 1), 0); del buf193  # reuse
        # Source Nodes: [x_180], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf197, addmm_69, 2420736, grid=grid(2420736), stream=stream0)
        del addmm_69
        buf198 = reinterpret_tensor(buf140, (1576, 384), (384, 1), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (1576, 1536), (1536, 1), 0), permute_361, out=buf198)
        del permute_361
        buf208 = reinterpret_tensor(buf130, (8, 197, 384), (75648, 384, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf198, primals_286, mul_179, buf192, div_35, buf208, 1576, 384, grid=grid(1576), stream=stream0)
        del div_35
        del primals_286
        buf209 = buf128; del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (1576, 384), (384, 1), 0), permute_365, out=buf209)
        del permute_365
        buf213 = reinterpret_tensor(buf116, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf209, buf213, 605184, grid=grid(605184), stream=stream0)
        buf214 = reinterpret_tensor(buf209, (48, 197, 64), (12608, 64, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_370, reinterpret_tensor(buf213, (48, 197, 64), (12608, 64, 1), 0), out=buf214)
        del permute_370
        buf220 = buf106; del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf214, buf220, 605184, grid=grid(605184), stream=stream0)
        buf222 = reinterpret_tensor(buf214, (1576, 384), (384, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf220, permute_378, out=buf222)
        del permute_378
        buf215 = reinterpret_tensor(buf125, (48, 197, 197), (38809, 197, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf213, (48, 197, 64), (12608, 64, 1), 0), permute_371, out=buf215)
        del permute_371
        buf217 = reinterpret_tensor(buf123, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_12.run(buf215, alias_28, buf217, 9456, 197, grid=grid(9456), stream=stream0)
        del alias_28
        buf218 = reinterpret_tensor(buf213, (48, 64, 197), (12608, 197, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_372, reinterpret_tensor(buf217, (48, 197, 197), (38809, 197, 1), 0), out=buf218)
        del permute_372
        buf219 = reinterpret_tensor(buf100, (48, 197, 64), (12608, 64, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf217, (48, 197, 197), (38809, 197, 1), 0), permute_373, out=buf219)
        del permute_373
        buf223 = buf131; del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf219, buf218, buf223, 18912, 64, grid=grid(18912, 64), stream=stream0)
        buf225 = reinterpret_tensor(buf219, (1576, 384), (384, 1), 0); del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (1576, 768), (768, 1), 0), permute_383, out=buf225)
        del permute_383
        buf232 = reinterpret_tensor(buf218, (8, 197, 384), (75648, 384, 1), 0); del buf218  # reuse
        # Source Nodes: [l__mod___blocks_9_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33.run(buf222, buf225, primals_280, cat_10, getitem_139, rsqrt_50, buf208, buf232, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_280
        buf233 = reinterpret_tensor(buf178, (1568, 384), (384, 1), 0); del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_17.run(buf232, buf233, 602112, grid=grid(602112), stream=stream0)
        buf234 = reinterpret_tensor(buf180, (1568, 384), (384, 1), 0); del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf233, permute_385, out=buf234)
        del permute_385
        buf244 = buf168; del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_34.run(buf244, buf182, buf185, primals_292, mul_174, buf234, primals_276, div_34, 25088, 24, grid=grid(25088), stream=stream0)
        del div_34
        del primals_276
        del primals_292
        buf188 = reinterpret_tensor(buf171, (24, 196), (1, 24), 0); del buf171  # reuse
        buf190 = buf164; del buf164  # reuse
        buf240 = buf98; del buf98  # reuse
        buf242 = buf96; del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_35.run(buf182, buf185, mul_174, buf234, buf188, buf190, buf240, buf242, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_174
        buf189 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf188, buf189, 24, 196, grid=grid(24), stream=stream0)
        buf191 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf190, buf191, 24, 196, grid=grid(24), stream=stream0)
        buf194 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf192, (384, 1576), (1, 384), 0), view_414, out=buf194)
        del view_414
        buf195 = buf144; del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf192, buf195, 4992, 122, grid=grid(4992), stream=stream0)
        buf196 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf195, buf196, 384, 13, grid=grid(384), stream=stream0)
        buf199 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (1536, 1576), (1, 1536), 0), view_412, out=buf199)
        del view_412
        buf200 = buf108; del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf197, buf200, 19968, 122, grid=grid(19968), stream=stream0)
        buf201 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf200, buf201, 1536, 13, grid=grid(1536), stream=stream0)
        buf204 = reinterpret_tensor(buf195, (384, 13), (1, 384), 0); del buf195  # reuse
        buf206 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf198, mul_179, buf204, buf206, 4992, 122, grid=grid(4992), stream=stream0)
        del mul_179
        buf205 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf204, buf205, 384, 13, grid=grid(384), stream=stream0)
        buf207 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf206, buf207, 384, 13, grid=grid(384), stream=stream0)
        buf210 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (384, 1576), (1, 384), 0), view_410, out=buf210)
        del view_410
        buf211 = reinterpret_tensor(buf206, (1, 384, 13), (4992, 1, 384), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf208, buf211, 4992, 122, grid=grid(4992), stream=stream0)
        buf212 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf211, buf212, 384, 13, grid=grid(384), stream=stream0)
        buf221 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf220, (384, 1576), (1, 384), 0), view_397, out=buf221)
        buf224 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (768, 1576), (1, 768), 0), view_397, out=buf224)
        del view_397
        buf228 = reinterpret_tensor(buf211, (384, 13), (1, 384), 0); del buf211  # reuse
        buf230 = buf204; del buf204  # reuse
        # Source Nodes: [l__mod___blocks_9_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf222, buf225, cat_10, getitem_139, rsqrt_50, buf228, buf230, 4992, 122, grid=grid(4992), stream=stream0)
        del cat_10
        del getitem_139
        del rsqrt_50
        buf229 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_9_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf228, buf229, 384, 13, grid=grid(384), stream=stream0)
        buf231 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf230, buf231, 384, 13, grid=grid(384), stream=stream0)
        buf235 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (384, 1568), (1, 384), 0), view_395, out=buf235)
        del view_395
        buf236 = reinterpret_tensor(buf230, (1, 384, 13), (4992, 1, 384), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf233, buf236, 4992, 121, grid=grid(4992), stream=stream0)
        buf237 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf236, buf237, 384, 13, grid=grid(384), stream=stream0)
        buf241 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf240, buf241, 24, 196, grid=grid(24), stream=stream0)
        buf243 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf242, buf243, 24, 196, grid=grid(24), stream=stream0)
        buf245 = reinterpret_tensor(buf157, (25088, 96), (96, 1), 0); del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf244, (25088, 24), (24, 1), 0), permute_389, out=buf245)
        del permute_389
        buf246 = empty((24, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf244, (24, 25088), (1, 24), 0), view_392, out=buf246)
        del view_392
        buf247 = reinterpret_tensor(buf242, (1, 24, 196), (4704, 1, 24), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf244, buf247, 4704, 128, grid=grid(4704), stream=stream0)
        buf248 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf247, buf248, 24, 196, grid=grid(24), stream=stream0)
        buf249 = reinterpret_tensor(buf245, (1568, 16, 96), (1536, 96, 1), 0); del buf245  # reuse
        # Source Nodes: [x_171], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf249, addmm_65, 2408448, grid=grid(2408448), stream=stream0)
        del addmm_65
        buf250 = reinterpret_tensor(buf233, (25088, 24), (24, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (25088, 96), (96, 1), 0), permute_393, out=buf250)
        del permute_393
        buf251 = empty((96, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (96, 25088), (1, 96), 0), view_390, out=buf251)
        del view_390
        buf252 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf249, buf252, 18816, 128, grid=grid(18816), stream=stream0)
        buf253 = empty((1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf252, buf253, 96, 196, grid=grid(96), stream=stream0)
        buf260 = buf244; del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_26.run(buf260, buf250, primals_270, mul_169, div_38, 25088, 24, grid=grid(25088), stream=stream0)
        del div_38
        del primals_270
        buf256 = reinterpret_tensor(buf247, (24, 196), (1, 24), 0); del buf247  # reuse
        buf258 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_20.run(buf250, mul_169, buf256, buf258, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_169
        buf257 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf256, buf257, 24, 196, grid=grid(24), stream=stream0)
        buf259 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf258, buf259, 24, 196, grid=grid(24), stream=stream0)
        buf261 = buf250; del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf260, (25088, 24), (24, 1), 0), permute_397, out=buf261)
        del permute_397
        buf262 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf260, (24, 25088), (1, 24), 0), view_388, out=buf262)
        del view_388
        buf263 = reinterpret_tensor(buf258, (1, 24, 196), (4704, 1, 24), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf260, buf263, 4704, 128, grid=grid(4704), stream=stream0)
        buf264 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf263, buf264, 24, 196, grid=grid(24), stream=stream0)
        buf265 = reinterpret_tensor(buf234, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf261, buf265, 602112, grid=grid(602112), stream=stream0)
        buf266 = reinterpret_tensor(buf261, (6272, 16, 6), (96, 6, 1), 0); del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_402, reinterpret_tensor(buf265, (6272, 16, 6), (96, 6, 1), 0), out=buf266)
        del permute_402
        buf267 = reinterpret_tensor(buf177, (6272, 16, 16), (256, 16, 1), 0); del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf265, (6272, 16, 6), (96, 6, 1), 0), permute_403, out=buf267)
        del permute_403
        buf269 = reinterpret_tensor(buf175, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_28.run(buf267, alias_29, buf269, 100352, 16, grid=grid(100352), stream=stream0)
        del alias_29
        buf270 = reinterpret_tensor(buf265, (6272, 6, 16), (96, 16, 1), 0); del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_404, reinterpret_tensor(buf269, (6272, 16, 16), (256, 16, 1), 0), out=buf270)
        del permute_404
        buf271 = reinterpret_tensor(buf185, (6272, 16, 6), (96, 6, 1), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf269, (6272, 16, 16), (256, 16, 1), 0), permute_405, out=buf271)
        del permute_405
        buf272 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_29.run(buf266, buf272, 602112, grid=grid(602112), stream=stream0)
        buf273 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf272, (24, 25088), (1, 24), 0), view_375, out=buf273)
        buf274 = reinterpret_tensor(buf266, (25088, 24), (24, 1), 0); del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf272, permute_410, out=buf274)
        del permute_410
        buf275 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf271, buf270, buf275, 200704, 6, grid=grid(200704, 6), stream=stream0)
        buf276 = empty((48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf275, (48, 25088), (1, 48), 0), view_375, out=buf276)
        del view_375
        buf277 = reinterpret_tensor(buf271, (25088, 24), (24, 1), 0); del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf275, (25088, 48), (48, 1), 0), permute_415, out=buf277)
        del permute_415
        buf284 = reinterpret_tensor(buf225, (8, 197, 384), (75648, 384, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_select_backward_slice_backward_31.run(buf232, buf284, 605184, grid=grid(605184), stream=stream0)
        buf285 = reinterpret_tensor(buf197, (1576, 1536), (1536, 1), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf284, (1576, 384), (384, 1), 0), permute_417, out=buf285)
        del permute_417
        buf289 = reinterpret_tensor(buf285, (8, 197, 1536), (302592, 1536, 1), 0); del buf285  # reuse
        # Source Nodes: [x_162], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf289, addmm_62, 2420736, grid=grid(2420736), stream=stream0)
        del addmm_62
        buf290 = reinterpret_tensor(buf232, (1576, 384), (384, 1), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (1576, 1536), (1536, 1), 0), permute_421, out=buf290)
        del permute_421
        buf300 = reinterpret_tensor(buf222, (8, 197, 384), (75648, 384, 1), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf290, primals_258, mul_161, buf284, div_40, buf300, 1576, 384, grid=grid(1576), stream=stream0)
        del div_40
        del primals_258
        buf301 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf300, (1576, 384), (384, 1), 0), permute_425, out=buf301)
        del permute_425
        buf305 = reinterpret_tensor(buf208, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf301, buf305, 605184, grid=grid(605184), stream=stream0)
        buf306 = reinterpret_tensor(buf301, (48, 197, 64), (12608, 64, 1), 0); del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_430, reinterpret_tensor(buf305, (48, 197, 64), (12608, 64, 1), 0), out=buf306)
        del permute_430
        buf312 = buf198; del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf306, buf312, 605184, grid=grid(605184), stream=stream0)
        buf314 = reinterpret_tensor(buf306, (1576, 384), (384, 1), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf312, permute_438, out=buf314)
        del permute_438
        buf307 = reinterpret_tensor(buf217, (48, 197, 197), (38809, 197, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf305, (48, 197, 64), (12608, 64, 1), 0), permute_431, out=buf307)
        del permute_431
        buf309 = reinterpret_tensor(buf215, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_12.run(buf307, alias_30, buf309, 9456, 197, grid=grid(9456), stream=stream0)
        del alias_30
        buf310 = reinterpret_tensor(buf305, (48, 64, 197), (12608, 197, 1), 0); del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_432, reinterpret_tensor(buf309, (48, 197, 197), (38809, 197, 1), 0), out=buf310)
        del permute_432
        buf311 = reinterpret_tensor(buf192, (48, 197, 64), (12608, 64, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf309, (48, 197, 197), (38809, 197, 1), 0), permute_433, out=buf311)
        del permute_433
        buf315 = buf223; del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf311, buf310, buf315, 18912, 64, grid=grid(18912, 64), stream=stream0)
        buf317 = reinterpret_tensor(buf311, (1576, 384), (384, 1), 0); del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (1576, 768), (768, 1), 0), permute_443, out=buf317)
        del permute_443
        buf324 = reinterpret_tensor(buf310, (8, 197, 384), (75648, 384, 1), 0); del buf310  # reuse
        # Source Nodes: [l__mod___blocks_8_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33.run(buf314, buf317, primals_252, cat_9, getitem_125, rsqrt_45, buf300, buf324, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_252
        buf325 = reinterpret_tensor(buf270, (1568, 384), (384, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_17.run(buf324, buf325, 602112, grid=grid(602112), stream=stream0)
        buf326 = reinterpret_tensor(buf272, (1568, 384), (384, 1), 0); del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf325, permute_445, out=buf326)
        del permute_445
        buf336 = buf260; del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_34.run(buf336, buf274, buf277, primals_264, mul_156, buf326, primals_248, div_39, 25088, 24, grid=grid(25088), stream=stream0)
        del div_39
        del primals_248
        del primals_264
        buf280 = reinterpret_tensor(buf263, (24, 196), (1, 24), 0); del buf263  # reuse
        buf282 = buf256; del buf256  # reuse
        buf332 = buf190; del buf190  # reuse
        buf334 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_35.run(buf274, buf277, mul_156, buf326, buf280, buf282, buf332, buf334, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_156
        buf281 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf280, buf281, 24, 196, grid=grid(24), stream=stream0)
        buf283 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf282, buf283, 24, 196, grid=grid(24), stream=stream0)
        buf286 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf284, (384, 1576), (1, 384), 0), view_373, out=buf286)
        del view_373
        buf287 = buf236; del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf284, buf287, 4992, 122, grid=grid(4992), stream=stream0)
        buf288 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf287, buf288, 384, 13, grid=grid(384), stream=stream0)
        buf291 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (1536, 1576), (1, 1536), 0), view_371, out=buf291)
        del view_371
        buf292 = buf200; del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf289, buf292, 19968, 122, grid=grid(19968), stream=stream0)
        buf293 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf292, buf293, 1536, 13, grid=grid(1536), stream=stream0)
        buf296 = reinterpret_tensor(buf287, (384, 13), (1, 384), 0); del buf287  # reuse
        buf298 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf290, mul_161, buf296, buf298, 4992, 122, grid=grid(4992), stream=stream0)
        del mul_161
        buf297 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf296, buf297, 384, 13, grid=grid(384), stream=stream0)
        buf299 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf298, buf299, 384, 13, grid=grid(384), stream=stream0)
        buf302 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf300, (384, 1576), (1, 384), 0), view_369, out=buf302)
        del view_369
        buf303 = reinterpret_tensor(buf298, (1, 384, 13), (4992, 1, 384), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf300, buf303, 4992, 122, grid=grid(4992), stream=stream0)
        buf304 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf303, buf304, 384, 13, grid=grid(384), stream=stream0)
        buf313 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf312, (384, 1576), (1, 384), 0), view_356, out=buf313)
        buf316 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (768, 1576), (1, 768), 0), view_356, out=buf316)
        del view_356
        buf320 = reinterpret_tensor(buf303, (384, 13), (1, 384), 0); del buf303  # reuse
        buf322 = buf296; del buf296  # reuse
        # Source Nodes: [l__mod___blocks_8_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf314, buf317, cat_9, getitem_125, rsqrt_45, buf320, buf322, 4992, 122, grid=grid(4992), stream=stream0)
        del cat_9
        del getitem_125
        del rsqrt_45
        buf321 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_8_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf320, buf321, 384, 13, grid=grid(384), stream=stream0)
        buf323 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf322, buf323, 384, 13, grid=grid(384), stream=stream0)
        buf327 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf325, (384, 1568), (1, 384), 0), view_354, out=buf327)
        del view_354
        buf328 = reinterpret_tensor(buf322, (1, 384, 13), (4992, 1, 384), 0); del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf325, buf328, 4992, 121, grid=grid(4992), stream=stream0)
        buf329 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf328, buf329, 384, 13, grid=grid(384), stream=stream0)
        buf333 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf332, buf333, 24, 196, grid=grid(24), stream=stream0)
        buf335 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf334, buf335, 24, 196, grid=grid(24), stream=stream0)
        buf337 = reinterpret_tensor(buf249, (25088, 96), (96, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf336, (25088, 24), (24, 1), 0), permute_449, out=buf337)
        del permute_449
        buf338 = empty((24, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf336, (24, 25088), (1, 24), 0), view_351, out=buf338)
        del view_351
        buf339 = reinterpret_tensor(buf334, (1, 24, 196), (4704, 1, 24), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf336, buf339, 4704, 128, grid=grid(4704), stream=stream0)
        buf340 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf339, buf340, 24, 196, grid=grid(24), stream=stream0)
        buf341 = reinterpret_tensor(buf337, (1568, 16, 96), (1536, 96, 1), 0); del buf337  # reuse
        # Source Nodes: [x_153], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf341, addmm_58, 2408448, grid=grid(2408448), stream=stream0)
        del addmm_58
        buf342 = reinterpret_tensor(buf325, (25088, 24), (24, 1), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf341, (25088, 96), (96, 1), 0), permute_453, out=buf342)
        del permute_453
        buf343 = empty((96, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf341, (96, 25088), (1, 96), 0), view_349, out=buf343)
        del view_349
        buf344 = buf252; del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf341, buf344, 18816, 128, grid=grid(18816), stream=stream0)
        buf345 = empty((1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf344, buf345, 96, 196, grid=grid(96), stream=stream0)
        buf352 = buf336; del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_26.run(buf352, buf342, primals_242, mul_151, div_43, 25088, 24, grid=grid(25088), stream=stream0)
        del div_43
        del primals_242
        buf348 = reinterpret_tensor(buf339, (24, 196), (1, 24), 0); del buf339  # reuse
        buf350 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_20.run(buf342, mul_151, buf348, buf350, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_151
        buf349 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf348, buf349, 24, 196, grid=grid(24), stream=stream0)
        buf351 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf350, buf351, 24, 196, grid=grid(24), stream=stream0)
        buf353 = buf342; del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf352, (25088, 24), (24, 1), 0), permute_457, out=buf353)
        del permute_457
        buf354 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf352, (24, 25088), (1, 24), 0), view_347, out=buf354)
        del view_347
        buf355 = reinterpret_tensor(buf350, (1, 24, 196), (4704, 1, 24), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf352, buf355, 4704, 128, grid=grid(4704), stream=stream0)
        buf356 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf355, buf356, 24, 196, grid=grid(24), stream=stream0)
        buf357 = reinterpret_tensor(buf326, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf353, buf357, 602112, grid=grid(602112), stream=stream0)
        buf358 = reinterpret_tensor(buf353, (6272, 16, 6), (96, 6, 1), 0); del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_462, reinterpret_tensor(buf357, (6272, 16, 6), (96, 6, 1), 0), out=buf358)
        del permute_462
        buf359 = reinterpret_tensor(buf269, (6272, 16, 16), (256, 16, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf357, (6272, 16, 6), (96, 6, 1), 0), permute_463, out=buf359)
        del permute_463
        buf361 = reinterpret_tensor(buf267, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_28.run(buf359, alias_31, buf361, 100352, 16, grid=grid(100352), stream=stream0)
        del alias_31
        buf362 = reinterpret_tensor(buf357, (6272, 6, 16), (96, 16, 1), 0); del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_464, reinterpret_tensor(buf361, (6272, 16, 16), (256, 16, 1), 0), out=buf362)
        del permute_464
        buf363 = reinterpret_tensor(buf277, (6272, 16, 6), (96, 6, 1), 0); del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf361, (6272, 16, 16), (256, 16, 1), 0), permute_465, out=buf363)
        del permute_465
        buf364 = buf274; del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_29.run(buf358, buf364, 602112, grid=grid(602112), stream=stream0)
        buf365 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf364, (24, 25088), (1, 24), 0), view_334, out=buf365)
        buf366 = reinterpret_tensor(buf358, (25088, 24), (24, 1), 0); del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf364, permute_470, out=buf366)
        del permute_470
        buf367 = buf275; del buf275  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf363, buf362, buf367, 200704, 6, grid=grid(200704, 6), stream=stream0)
        buf368 = empty((48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf367, (48, 25088), (1, 48), 0), view_334, out=buf368)
        del view_334
        buf369 = reinterpret_tensor(buf363, (25088, 24), (24, 1), 0); del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf367, (25088, 48), (48, 1), 0), permute_475, out=buf369)
        del permute_475
        buf376 = reinterpret_tensor(buf317, (8, 197, 384), (75648, 384, 1), 0); del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_select_backward_slice_backward_31.run(buf324, buf376, 605184, grid=grid(605184), stream=stream0)
        buf377 = reinterpret_tensor(buf289, (1576, 1536), (1536, 1), 0); del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf376, (1576, 384), (384, 1), 0), permute_477, out=buf377)
        del permute_477
        buf381 = reinterpret_tensor(buf377, (8, 197, 1536), (302592, 1536, 1), 0); del buf377  # reuse
        # Source Nodes: [x_144], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf381, addmm_55, 2420736, grid=grid(2420736), stream=stream0)
        del addmm_55
        buf382 = reinterpret_tensor(buf324, (1576, 384), (384, 1), 0); del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf381, (1576, 1536), (1536, 1), 0), permute_481, out=buf382)
        del permute_481
        buf392 = reinterpret_tensor(buf314, (8, 197, 384), (75648, 384, 1), 0); del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf382, primals_230, mul_143, buf376, div_45, buf392, 1576, 384, grid=grid(1576), stream=stream0)
        del div_45
        del primals_230
        buf393 = buf312; del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf392, (1576, 384), (384, 1), 0), permute_485, out=buf393)
        del permute_485
        buf397 = reinterpret_tensor(buf300, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf393, buf397, 605184, grid=grid(605184), stream=stream0)
        buf398 = reinterpret_tensor(buf393, (48, 197, 64), (12608, 64, 1), 0); del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_490, reinterpret_tensor(buf397, (48, 197, 64), (12608, 64, 1), 0), out=buf398)
        del permute_490
        buf404 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf398, buf404, 605184, grid=grid(605184), stream=stream0)
        buf406 = reinterpret_tensor(buf398, (1576, 384), (384, 1), 0); del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf404, permute_498, out=buf406)
        del permute_498
        buf399 = reinterpret_tensor(buf309, (48, 197, 197), (38809, 197, 1), 0); del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf397, (48, 197, 64), (12608, 64, 1), 0), permute_491, out=buf399)
        del permute_491
        buf401 = reinterpret_tensor(buf307, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_12.run(buf399, alias_32, buf401, 9456, 197, grid=grid(9456), stream=stream0)
        del alias_32
        buf402 = reinterpret_tensor(buf397, (48, 64, 197), (12608, 197, 1), 0); del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_492, reinterpret_tensor(buf401, (48, 197, 197), (38809, 197, 1), 0), out=buf402)
        del permute_492
        buf403 = reinterpret_tensor(buf284, (48, 197, 64), (12608, 64, 1), 0); del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (48, 197, 197), (38809, 197, 1), 0), permute_493, out=buf403)
        del permute_493
        buf407 = buf315; del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf403, buf402, buf407, 18912, 64, grid=grid(18912, 64), stream=stream0)
        buf409 = reinterpret_tensor(buf403, (1576, 384), (384, 1), 0); del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf407, (1576, 768), (768, 1), 0), permute_503, out=buf409)
        del permute_503
        buf416 = reinterpret_tensor(buf402, (8, 197, 384), (75648, 384, 1), 0); del buf402  # reuse
        # Source Nodes: [l__mod___blocks_7_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33.run(buf406, buf409, primals_224, cat_8, getitem_111, rsqrt_40, buf392, buf416, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_224
        buf417 = reinterpret_tensor(buf362, (1568, 384), (384, 1), 0); del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_17.run(buf416, buf417, 602112, grid=grid(602112), stream=stream0)
        buf418 = reinterpret_tensor(buf364, (1568, 384), (384, 1), 0); del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf417, permute_505, out=buf418)
        del permute_505
        buf428 = buf352; del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_34.run(buf428, buf366, buf369, primals_236, mul_138, buf418, primals_220, div_44, 25088, 24, grid=grid(25088), stream=stream0)
        del div_44
        del primals_220
        del primals_236
        buf372 = reinterpret_tensor(buf355, (24, 196), (1, 24), 0); del buf355  # reuse
        buf374 = buf348; del buf348  # reuse
        buf424 = buf282; del buf282  # reuse
        buf426 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_35.run(buf366, buf369, mul_138, buf418, buf372, buf374, buf424, buf426, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_138
        buf373 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf372, buf373, 24, 196, grid=grid(24), stream=stream0)
        buf375 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf374, buf375, 24, 196, grid=grid(24), stream=stream0)
        buf378 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf376, (384, 1576), (1, 384), 0), view_332, out=buf378)
        del view_332
        buf379 = buf328; del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf376, buf379, 4992, 122, grid=grid(4992), stream=stream0)
        buf380 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf379, buf380, 384, 13, grid=grid(384), stream=stream0)
        buf383 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf381, (1536, 1576), (1, 1536), 0), view_330, out=buf383)
        del view_330
        buf384 = buf292; del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf381, buf384, 19968, 122, grid=grid(19968), stream=stream0)
        buf385 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf384, buf385, 1536, 13, grid=grid(1536), stream=stream0)
        buf388 = reinterpret_tensor(buf379, (384, 13), (1, 384), 0); del buf379  # reuse
        buf390 = buf320; del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf382, mul_143, buf388, buf390, 4992, 122, grid=grid(4992), stream=stream0)
        del mul_143
        buf389 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf388, buf389, 384, 13, grid=grid(384), stream=stream0)
        buf391 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf390, buf391, 384, 13, grid=grid(384), stream=stream0)
        buf394 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf392, (384, 1576), (1, 384), 0), view_328, out=buf394)
        del view_328
        buf395 = reinterpret_tensor(buf390, (1, 384, 13), (4992, 1, 384), 0); del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf392, buf395, 4992, 122, grid=grid(4992), stream=stream0)
        buf396 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf395, buf396, 384, 13, grid=grid(384), stream=stream0)
        buf405 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf404, (384, 1576), (1, 384), 0), view_315, out=buf405)
        buf408 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf407, (768, 1576), (1, 768), 0), view_315, out=buf408)
        del view_315
        buf412 = reinterpret_tensor(buf395, (384, 13), (1, 384), 0); del buf395  # reuse
        buf414 = buf388; del buf388  # reuse
        # Source Nodes: [l__mod___blocks_7_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf406, buf409, cat_8, getitem_111, rsqrt_40, buf412, buf414, 4992, 122, grid=grid(4992), stream=stream0)
        del cat_8
        del getitem_111
        del rsqrt_40
        buf413 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_7_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf412, buf413, 384, 13, grid=grid(384), stream=stream0)
        buf415 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf414, buf415, 384, 13, grid=grid(384), stream=stream0)
        buf419 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf417, (384, 1568), (1, 384), 0), view_313, out=buf419)
        del view_313
        buf420 = reinterpret_tensor(buf414, (1, 384, 13), (4992, 1, 384), 0); del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf417, buf420, 4992, 121, grid=grid(4992), stream=stream0)
        buf421 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf420, buf421, 384, 13, grid=grid(384), stream=stream0)
        buf425 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf424, buf425, 24, 196, grid=grid(24), stream=stream0)
        buf427 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf426, buf427, 24, 196, grid=grid(24), stream=stream0)
        buf429 = reinterpret_tensor(buf341, (25088, 96), (96, 1), 0); del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (25088, 24), (24, 1), 0), permute_509, out=buf429)
        del permute_509
        buf430 = empty((24, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (24, 25088), (1, 24), 0), view_310, out=buf430)
        del view_310
        buf431 = reinterpret_tensor(buf426, (1, 24, 196), (4704, 1, 24), 0); del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf428, buf431, 4704, 128, grid=grid(4704), stream=stream0)
        buf432 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf431, buf432, 24, 196, grid=grid(24), stream=stream0)
        buf433 = reinterpret_tensor(buf429, (1568, 16, 96), (1536, 96, 1), 0); del buf429  # reuse
        # Source Nodes: [x_135], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf433, addmm_51, 2408448, grid=grid(2408448), stream=stream0)
        del addmm_51
        buf434 = reinterpret_tensor(buf417, (25088, 24), (24, 1), 0); del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf433, (25088, 96), (96, 1), 0), permute_513, out=buf434)
        del permute_513
        buf435 = empty((96, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf433, (96, 25088), (1, 96), 0), view_308, out=buf435)
        del view_308
        buf436 = buf344; del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf433, buf436, 18816, 128, grid=grid(18816), stream=stream0)
        buf437 = empty((1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf436, buf437, 96, 196, grid=grid(96), stream=stream0)
        buf444 = buf428; del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_26.run(buf444, buf434, primals_214, mul_133, div_48, 25088, 24, grid=grid(25088), stream=stream0)
        del div_48
        del primals_214
        buf440 = reinterpret_tensor(buf431, (24, 196), (1, 24), 0); del buf431  # reuse
        buf442 = buf424; del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_20.run(buf434, mul_133, buf440, buf442, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_133
        buf441 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf440, buf441, 24, 196, grid=grid(24), stream=stream0)
        buf443 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf442, buf443, 24, 196, grid=grid(24), stream=stream0)
        buf445 = buf434; del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf444, (25088, 24), (24, 1), 0), permute_517, out=buf445)
        del permute_517
        buf446 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf444, (24, 25088), (1, 24), 0), view_306, out=buf446)
        del view_306
        buf447 = reinterpret_tensor(buf442, (1, 24, 196), (4704, 1, 24), 0); del buf442  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf444, buf447, 4704, 128, grid=grid(4704), stream=stream0)
        buf448 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf447, buf448, 24, 196, grid=grid(24), stream=stream0)
        buf449 = reinterpret_tensor(buf418, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf445, buf449, 602112, grid=grid(602112), stream=stream0)
        buf450 = reinterpret_tensor(buf445, (6272, 16, 6), (96, 6, 1), 0); del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_522, reinterpret_tensor(buf449, (6272, 16, 6), (96, 6, 1), 0), out=buf450)
        del permute_522
        buf451 = reinterpret_tensor(buf361, (6272, 16, 16), (256, 16, 1), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf449, (6272, 16, 6), (96, 6, 1), 0), permute_523, out=buf451)
        del permute_523
        buf453 = reinterpret_tensor(buf359, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_28.run(buf451, alias_33, buf453, 100352, 16, grid=grid(100352), stream=stream0)
        del alias_33
        buf454 = reinterpret_tensor(buf449, (6272, 6, 16), (96, 16, 1), 0); del buf449  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_524, reinterpret_tensor(buf453, (6272, 16, 16), (256, 16, 1), 0), out=buf454)
        del permute_524
        buf455 = reinterpret_tensor(buf369, (6272, 16, 6), (96, 6, 1), 0); del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf453, (6272, 16, 16), (256, 16, 1), 0), permute_525, out=buf455)
        del permute_525
        buf456 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_29.run(buf450, buf456, 602112, grid=grid(602112), stream=stream0)
        buf457 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf456, (24, 25088), (1, 24), 0), view_293, out=buf457)
        buf458 = reinterpret_tensor(buf450, (25088, 24), (24, 1), 0); del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf456, permute_530, out=buf458)
        del permute_530
        buf459 = buf367; del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf455, buf454, buf459, 200704, 6, grid=grid(200704, 6), stream=stream0)
        buf460 = empty((48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf459, (48, 25088), (1, 48), 0), view_293, out=buf460)
        del view_293
        buf461 = reinterpret_tensor(buf455, (25088, 24), (24, 1), 0); del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf459, (25088, 48), (48, 1), 0), permute_535, out=buf461)
        del permute_535
        buf468 = reinterpret_tensor(buf409, (8, 197, 384), (75648, 384, 1), 0); del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_select_backward_slice_backward_31.run(buf416, buf468, 605184, grid=grid(605184), stream=stream0)
        buf469 = reinterpret_tensor(buf381, (1576, 1536), (1536, 1), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf468, (1576, 384), (384, 1), 0), permute_537, out=buf469)
        del permute_537
        buf473 = reinterpret_tensor(buf469, (8, 197, 1536), (302592, 1536, 1), 0); del buf469  # reuse
        # Source Nodes: [x_126], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf473, addmm_48, 2420736, grid=grid(2420736), stream=stream0)
        del addmm_48
        buf474 = reinterpret_tensor(buf416, (1576, 384), (384, 1), 0); del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf473, (1576, 1536), (1536, 1), 0), permute_541, out=buf474)
        del permute_541
        buf484 = reinterpret_tensor(buf406, (8, 197, 384), (75648, 384, 1), 0); del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf474, primals_202, mul_125, buf468, div_50, buf484, 1576, 384, grid=grid(1576), stream=stream0)
        del div_50
        del primals_202
        buf485 = buf404; del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (1576, 384), (384, 1), 0), permute_545, out=buf485)
        del permute_545
        buf489 = reinterpret_tensor(buf392, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf485, buf489, 605184, grid=grid(605184), stream=stream0)
        buf490 = reinterpret_tensor(buf485, (48, 197, 64), (12608, 64, 1), 0); del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_550, reinterpret_tensor(buf489, (48, 197, 64), (12608, 64, 1), 0), out=buf490)
        del permute_550
        buf496 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf490, buf496, 605184, grid=grid(605184), stream=stream0)
        buf498 = reinterpret_tensor(buf490, (1576, 384), (384, 1), 0); del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf496, permute_558, out=buf498)
        del permute_558
        buf491 = reinterpret_tensor(buf401, (48, 197, 197), (38809, 197, 1), 0); del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf489, (48, 197, 64), (12608, 64, 1), 0), permute_551, out=buf491)
        del permute_551
        buf493 = reinterpret_tensor(buf399, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_12.run(buf491, alias_34, buf493, 9456, 197, grid=grid(9456), stream=stream0)
        del alias_34
        buf494 = reinterpret_tensor(buf489, (48, 64, 197), (12608, 197, 1), 0); del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_552, reinterpret_tensor(buf493, (48, 197, 197), (38809, 197, 1), 0), out=buf494)
        del permute_552
        buf495 = reinterpret_tensor(buf376, (48, 197, 64), (12608, 64, 1), 0); del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf493, (48, 197, 197), (38809, 197, 1), 0), permute_553, out=buf495)
        del permute_553
        buf499 = buf407; del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf495, buf494, buf499, 18912, 64, grid=grid(18912, 64), stream=stream0)
        buf501 = reinterpret_tensor(buf495, (1576, 384), (384, 1), 0); del buf495  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf499, (1576, 768), (768, 1), 0), permute_563, out=buf501)
        del permute_563
        buf508 = reinterpret_tensor(buf494, (8, 197, 384), (75648, 384, 1), 0); del buf494  # reuse
        # Source Nodes: [l__mod___blocks_6_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33.run(buf498, buf501, primals_196, cat_7, getitem_97, rsqrt_35, buf484, buf508, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_196
        buf509 = reinterpret_tensor(buf454, (1568, 384), (384, 1), 0); del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_17.run(buf508, buf509, 602112, grid=grid(602112), stream=stream0)
        buf510 = reinterpret_tensor(buf456, (1568, 384), (384, 1), 0); del buf456  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf509, permute_565, out=buf510)
        del permute_565
        buf520 = buf444; del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_34.run(buf520, buf458, buf461, primals_208, mul_120, buf510, primals_192, div_49, 25088, 24, grid=grid(25088), stream=stream0)
        del div_49
        del primals_192
        del primals_208
        buf464 = reinterpret_tensor(buf447, (24, 196), (1, 24), 0); del buf447  # reuse
        buf466 = buf440; del buf440  # reuse
        buf516 = buf374; del buf374  # reuse
        buf518 = buf372; del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_35.run(buf458, buf461, mul_120, buf510, buf464, buf466, buf516, buf518, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_120
        buf465 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf464, buf465, 24, 196, grid=grid(24), stream=stream0)
        buf467 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf466, buf467, 24, 196, grid=grid(24), stream=stream0)
        buf470 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf468, (384, 1576), (1, 384), 0), view_291, out=buf470)
        del view_291
        buf471 = buf420; del buf420  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf468, buf471, 4992, 122, grid=grid(4992), stream=stream0)
        buf472 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf471, buf472, 384, 13, grid=grid(384), stream=stream0)
        buf475 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf473, (1536, 1576), (1, 1536), 0), view_289, out=buf475)
        del view_289
        buf476 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf473, buf476, 19968, 122, grid=grid(19968), stream=stream0)
        buf477 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf476, buf477, 1536, 13, grid=grid(1536), stream=stream0)
        buf480 = reinterpret_tensor(buf471, (384, 13), (1, 384), 0); del buf471  # reuse
        buf482 = buf412; del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf474, mul_125, buf480, buf482, 4992, 122, grid=grid(4992), stream=stream0)
        del mul_125
        buf481 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf480, buf481, 384, 13, grid=grid(384), stream=stream0)
        buf483 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf482, buf483, 384, 13, grid=grid(384), stream=stream0)
        buf486 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (384, 1576), (1, 384), 0), view_287, out=buf486)
        del view_287
        buf487 = reinterpret_tensor(buf482, (1, 384, 13), (4992, 1, 384), 0); del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf484, buf487, 4992, 122, grid=grid(4992), stream=stream0)
        buf488 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf487, buf488, 384, 13, grid=grid(384), stream=stream0)
        buf497 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf496, (384, 1576), (1, 384), 0), view_274, out=buf497)
        buf500 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf499, (768, 1576), (1, 768), 0), view_274, out=buf500)
        del view_274
        buf504 = reinterpret_tensor(buf487, (384, 13), (1, 384), 0); del buf487  # reuse
        buf506 = buf480; del buf480  # reuse
        # Source Nodes: [l__mod___blocks_6_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf498, buf501, cat_7, getitem_97, rsqrt_35, buf504, buf506, 4992, 122, grid=grid(4992), stream=stream0)
        del cat_7
        del getitem_97
        del rsqrt_35
        buf505 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_6_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf504, buf505, 384, 13, grid=grid(384), stream=stream0)
        buf507 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf506, buf507, 384, 13, grid=grid(384), stream=stream0)
        buf511 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf509, (384, 1568), (1, 384), 0), view_272, out=buf511)
        del view_272
        buf512 = reinterpret_tensor(buf506, (1, 384, 13), (4992, 1, 384), 0); del buf506  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf509, buf512, 4992, 121, grid=grid(4992), stream=stream0)
        buf513 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf512, buf513, 384, 13, grid=grid(384), stream=stream0)
        buf517 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf516, buf517, 24, 196, grid=grid(24), stream=stream0)
        buf519 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf518, buf519, 24, 196, grid=grid(24), stream=stream0)
        buf521 = reinterpret_tensor(buf433, (25088, 96), (96, 1), 0); del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf520, (25088, 24), (24, 1), 0), permute_569, out=buf521)
        del permute_569
        buf522 = empty((24, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf520, (24, 25088), (1, 24), 0), view_269, out=buf522)
        del view_269
        buf523 = reinterpret_tensor(buf518, (1, 24, 196), (4704, 1, 24), 0); del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf520, buf523, 4704, 128, grid=grid(4704), stream=stream0)
        buf524 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf523, buf524, 24, 196, grid=grid(24), stream=stream0)
        buf525 = reinterpret_tensor(buf521, (1568, 16, 96), (1536, 96, 1), 0); del buf521  # reuse
        # Source Nodes: [x_117], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf525, addmm_44, 2408448, grid=grid(2408448), stream=stream0)
        del addmm_44
        buf526 = reinterpret_tensor(buf509, (25088, 24), (24, 1), 0); del buf509  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf525, (25088, 96), (96, 1), 0), permute_573, out=buf526)
        del permute_573
        buf527 = empty((96, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf525, (96, 25088), (1, 96), 0), view_267, out=buf527)
        del view_267
        buf528 = buf436; del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf525, buf528, 18816, 128, grid=grid(18816), stream=stream0)
        buf529 = empty((1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf528, buf529, 96, 196, grid=grid(96), stream=stream0)
        buf536 = buf520; del buf520  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_26.run(buf536, buf526, primals_186, mul_115, div_53, 25088, 24, grid=grid(25088), stream=stream0)
        del div_53
        del primals_186
        buf532 = reinterpret_tensor(buf523, (24, 196), (1, 24), 0); del buf523  # reuse
        buf534 = buf516; del buf516  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_20.run(buf526, mul_115, buf532, buf534, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_115
        buf533 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf532, buf533, 24, 196, grid=grid(24), stream=stream0)
        buf535 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf534, buf535, 24, 196, grid=grid(24), stream=stream0)
        buf537 = buf526; del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf536, (25088, 24), (24, 1), 0), permute_577, out=buf537)
        del permute_577
        buf538 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf536, (24, 25088), (1, 24), 0), view_265, out=buf538)
        del view_265
        buf539 = reinterpret_tensor(buf534, (1, 24, 196), (4704, 1, 24), 0); del buf534  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf536, buf539, 4704, 128, grid=grid(4704), stream=stream0)
        buf540 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf539, buf540, 24, 196, grid=grid(24), stream=stream0)
        buf541 = reinterpret_tensor(buf510, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf537, buf541, 602112, grid=grid(602112), stream=stream0)
        buf542 = reinterpret_tensor(buf537, (6272, 16, 6), (96, 6, 1), 0); del buf537  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_582, reinterpret_tensor(buf541, (6272, 16, 6), (96, 6, 1), 0), out=buf542)
        del permute_582
        buf543 = reinterpret_tensor(buf453, (6272, 16, 16), (256, 16, 1), 0); del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf541, (6272, 16, 6), (96, 6, 1), 0), permute_583, out=buf543)
        del permute_583
        buf545 = reinterpret_tensor(buf451, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf451  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_28.run(buf543, alias_35, buf545, 100352, 16, grid=grid(100352), stream=stream0)
        del alias_35
        buf546 = reinterpret_tensor(buf541, (6272, 6, 16), (96, 16, 1), 0); del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_584, reinterpret_tensor(buf545, (6272, 16, 16), (256, 16, 1), 0), out=buf546)
        del permute_584
        buf547 = reinterpret_tensor(buf461, (6272, 16, 6), (96, 6, 1), 0); del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf545, (6272, 16, 16), (256, 16, 1), 0), permute_585, out=buf547)
        del permute_585
        buf548 = buf458; del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_29.run(buf542, buf548, 602112, grid=grid(602112), stream=stream0)
        buf549 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf548, (24, 25088), (1, 24), 0), view_252, out=buf549)
        buf550 = reinterpret_tensor(buf542, (25088, 24), (24, 1), 0); del buf542  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf548, permute_590, out=buf550)
        del permute_590
        buf551 = buf459; del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf547, buf546, buf551, 200704, 6, grid=grid(200704, 6), stream=stream0)
        buf552 = empty((48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf551, (48, 25088), (1, 48), 0), view_252, out=buf552)
        del view_252
        buf553 = reinterpret_tensor(buf547, (25088, 24), (24, 1), 0); del buf547  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf551, (25088, 48), (48, 1), 0), permute_595, out=buf553)
        del permute_595
        buf560 = reinterpret_tensor(buf501, (8, 197, 384), (75648, 384, 1), 0); del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_select_backward_slice_backward_31.run(buf508, buf560, 605184, grid=grid(605184), stream=stream0)
        buf561 = reinterpret_tensor(buf473, (1576, 1536), (1536, 1), 0); del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf560, (1576, 384), (384, 1), 0), permute_597, out=buf561)
        del permute_597
        buf565 = reinterpret_tensor(buf561, (8, 197, 1536), (302592, 1536, 1), 0); del buf561  # reuse
        # Source Nodes: [x_108], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf565, addmm_41, 2420736, grid=grid(2420736), stream=stream0)
        del addmm_41
        buf566 = reinterpret_tensor(buf508, (1576, 384), (384, 1), 0); del buf508  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf565, (1576, 1536), (1536, 1), 0), permute_601, out=buf566)
        del permute_601
        buf576 = reinterpret_tensor(buf498, (8, 197, 384), (75648, 384, 1), 0); del buf498  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf566, primals_174, mul_107, buf560, div_55, buf576, 1576, 384, grid=grid(1576), stream=stream0)
        del div_55
        del primals_174
        buf577 = buf496; del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf576, (1576, 384), (384, 1), 0), permute_605, out=buf577)
        del permute_605
        buf581 = reinterpret_tensor(buf484, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf484  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf577, buf581, 605184, grid=grid(605184), stream=stream0)
        buf582 = reinterpret_tensor(buf577, (48, 197, 64), (12608, 64, 1), 0); del buf577  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_610, reinterpret_tensor(buf581, (48, 197, 64), (12608, 64, 1), 0), out=buf582)
        del permute_610
        buf588 = buf474; del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf582, buf588, 605184, grid=grid(605184), stream=stream0)
        buf590 = reinterpret_tensor(buf582, (1576, 384), (384, 1), 0); del buf582  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf588, permute_618, out=buf590)
        del permute_618
        buf583 = reinterpret_tensor(buf493, (48, 197, 197), (38809, 197, 1), 0); del buf493  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf581, (48, 197, 64), (12608, 64, 1), 0), permute_611, out=buf583)
        del permute_611
        buf585 = reinterpret_tensor(buf491, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf491  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_12.run(buf583, alias_36, buf585, 9456, 197, grid=grid(9456), stream=stream0)
        del alias_36
        buf586 = reinterpret_tensor(buf581, (48, 64, 197), (12608, 197, 1), 0); del buf581  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_612, reinterpret_tensor(buf585, (48, 197, 197), (38809, 197, 1), 0), out=buf586)
        del permute_612
        buf587 = reinterpret_tensor(buf468, (48, 197, 64), (12608, 64, 1), 0); del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf585, (48, 197, 197), (38809, 197, 1), 0), permute_613, out=buf587)
        del permute_613
        buf591 = buf499; del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf587, buf586, buf591, 18912, 64, grid=grid(18912, 64), stream=stream0)
        buf593 = reinterpret_tensor(buf587, (1576, 384), (384, 1), 0); del buf587  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf591, (1576, 768), (768, 1), 0), permute_623, out=buf593)
        del permute_623
        buf600 = reinterpret_tensor(buf586, (8, 197, 384), (75648, 384, 1), 0); del buf586  # reuse
        # Source Nodes: [l__mod___blocks_5_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33.run(buf590, buf593, primals_168, cat_6, getitem_83, rsqrt_30, buf576, buf600, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_168
        buf601 = reinterpret_tensor(buf546, (1568, 384), (384, 1), 0); del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_17.run(buf600, buf601, 602112, grid=grid(602112), stream=stream0)
        buf602 = reinterpret_tensor(buf548, (1568, 384), (384, 1), 0); del buf548  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf601, permute_625, out=buf602)
        del permute_625
        buf612 = buf536; del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_34.run(buf612, buf550, buf553, primals_180, mul_102, buf602, primals_164, div_54, 25088, 24, grid=grid(25088), stream=stream0)
        del div_54
        del primals_164
        del primals_180
        buf556 = reinterpret_tensor(buf539, (24, 196), (1, 24), 0); del buf539  # reuse
        buf558 = buf532; del buf532  # reuse
        buf608 = buf466; del buf466  # reuse
        buf610 = buf464; del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_35.run(buf550, buf553, mul_102, buf602, buf556, buf558, buf608, buf610, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_102
        buf557 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf556, buf557, 24, 196, grid=grid(24), stream=stream0)
        buf559 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf558, buf559, 24, 196, grid=grid(24), stream=stream0)
        buf562 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf560, (384, 1576), (1, 384), 0), view_250, out=buf562)
        del view_250
        buf563 = buf512; del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf560, buf563, 4992, 122, grid=grid(4992), stream=stream0)
        buf564 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf563, buf564, 384, 13, grid=grid(384), stream=stream0)
        buf567 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf565, (1536, 1576), (1, 1536), 0), view_248, out=buf567)
        del view_248
        buf568 = buf476; del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf565, buf568, 19968, 122, grid=grid(19968), stream=stream0)
        buf569 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf568, buf569, 1536, 13, grid=grid(1536), stream=stream0)
        buf572 = reinterpret_tensor(buf563, (384, 13), (1, 384), 0); del buf563  # reuse
        buf574 = buf504; del buf504  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf566, mul_107, buf572, buf574, 4992, 122, grid=grid(4992), stream=stream0)
        del mul_107
        buf573 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf572, buf573, 384, 13, grid=grid(384), stream=stream0)
        buf575 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf574, buf575, 384, 13, grid=grid(384), stream=stream0)
        buf578 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf576, (384, 1576), (1, 384), 0), view_246, out=buf578)
        del view_246
        buf579 = reinterpret_tensor(buf574, (1, 384, 13), (4992, 1, 384), 0); del buf574  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf576, buf579, 4992, 122, grid=grid(4992), stream=stream0)
        buf580 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf579, buf580, 384, 13, grid=grid(384), stream=stream0)
        buf589 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf588, (384, 1576), (1, 384), 0), view_233, out=buf589)
        buf592 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf591, (768, 1576), (1, 768), 0), view_233, out=buf592)
        del view_233
        buf596 = reinterpret_tensor(buf579, (384, 13), (1, 384), 0); del buf579  # reuse
        buf598 = buf572; del buf572  # reuse
        # Source Nodes: [l__mod___blocks_5_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf590, buf593, cat_6, getitem_83, rsqrt_30, buf596, buf598, 4992, 122, grid=grid(4992), stream=stream0)
        del cat_6
        del getitem_83
        del rsqrt_30
        buf597 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_5_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf596, buf597, 384, 13, grid=grid(384), stream=stream0)
        buf599 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf598, buf599, 384, 13, grid=grid(384), stream=stream0)
        buf603 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf601, (384, 1568), (1, 384), 0), view_231, out=buf603)
        del view_231
        buf604 = reinterpret_tensor(buf598, (1, 384, 13), (4992, 1, 384), 0); del buf598  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf601, buf604, 4992, 121, grid=grid(4992), stream=stream0)
        buf605 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf604, buf605, 384, 13, grid=grid(384), stream=stream0)
        buf609 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf608, buf609, 24, 196, grid=grid(24), stream=stream0)
        buf611 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf610, buf611, 24, 196, grid=grid(24), stream=stream0)
        buf613 = reinterpret_tensor(buf525, (25088, 96), (96, 1), 0); del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf612, (25088, 24), (24, 1), 0), permute_629, out=buf613)
        del permute_629
        buf614 = empty((24, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf612, (24, 25088), (1, 24), 0), view_228, out=buf614)
        del view_228
        buf615 = reinterpret_tensor(buf610, (1, 24, 196), (4704, 1, 24), 0); del buf610  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf612, buf615, 4704, 128, grid=grid(4704), stream=stream0)
        buf616 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf615, buf616, 24, 196, grid=grid(24), stream=stream0)
        buf617 = reinterpret_tensor(buf613, (1568, 16, 96), (1536, 96, 1), 0); del buf613  # reuse
        # Source Nodes: [x_99], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf617, addmm_37, 2408448, grid=grid(2408448), stream=stream0)
        del addmm_37
        buf618 = reinterpret_tensor(buf601, (25088, 24), (24, 1), 0); del buf601  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf617, (25088, 96), (96, 1), 0), permute_633, out=buf618)
        del permute_633
        buf619 = empty((96, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf617, (96, 25088), (1, 96), 0), view_226, out=buf619)
        del view_226
        buf620 = buf528; del buf528  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf617, buf620, 18816, 128, grid=grid(18816), stream=stream0)
        buf621 = empty((1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf620, buf621, 96, 196, grid=grid(96), stream=stream0)
        buf628 = buf612; del buf612  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_26.run(buf628, buf618, primals_158, mul_97, div_58, 25088, 24, grid=grid(25088), stream=stream0)
        del div_58
        del primals_158
        buf624 = reinterpret_tensor(buf615, (24, 196), (1, 24), 0); del buf615  # reuse
        buf626 = buf608; del buf608  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_20.run(buf618, mul_97, buf624, buf626, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_97
        buf625 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf624, buf625, 24, 196, grid=grid(24), stream=stream0)
        buf627 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf626, buf627, 24, 196, grid=grid(24), stream=stream0)
        buf629 = buf618; del buf618  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf628, (25088, 24), (24, 1), 0), permute_637, out=buf629)
        del permute_637
        buf630 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf628, (24, 25088), (1, 24), 0), view_224, out=buf630)
        del view_224
        buf631 = reinterpret_tensor(buf626, (1, 24, 196), (4704, 1, 24), 0); del buf626  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf628, buf631, 4704, 128, grid=grid(4704), stream=stream0)
        buf632 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf631, buf632, 24, 196, grid=grid(24), stream=stream0)
        buf633 = reinterpret_tensor(buf602, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf602  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf629, buf633, 602112, grid=grid(602112), stream=stream0)
        buf634 = reinterpret_tensor(buf629, (6272, 16, 6), (96, 6, 1), 0); del buf629  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_642, reinterpret_tensor(buf633, (6272, 16, 6), (96, 6, 1), 0), out=buf634)
        del permute_642
        buf635 = reinterpret_tensor(buf545, (6272, 16, 16), (256, 16, 1), 0); del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf633, (6272, 16, 6), (96, 6, 1), 0), permute_643, out=buf635)
        del permute_643
        buf637 = reinterpret_tensor(buf543, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf543  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_28.run(buf635, alias_37, buf637, 100352, 16, grid=grid(100352), stream=stream0)
        del alias_37
        buf638 = reinterpret_tensor(buf633, (6272, 6, 16), (96, 16, 1), 0); del buf633  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_644, reinterpret_tensor(buf637, (6272, 16, 16), (256, 16, 1), 0), out=buf638)
        del permute_644
        buf639 = reinterpret_tensor(buf553, (6272, 16, 6), (96, 6, 1), 0); del buf553  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf637, (6272, 16, 16), (256, 16, 1), 0), permute_645, out=buf639)
        del permute_645
        buf640 = buf550; del buf550  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_29.run(buf634, buf640, 602112, grid=grid(602112), stream=stream0)
        buf641 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf640, (24, 25088), (1, 24), 0), view_211, out=buf641)
        buf642 = reinterpret_tensor(buf634, (25088, 24), (24, 1), 0); del buf634  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf640, permute_650, out=buf642)
        del permute_650
        buf643 = buf551; del buf551  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf639, buf638, buf643, 200704, 6, grid=grid(200704, 6), stream=stream0)
        buf644 = empty((48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf643, (48, 25088), (1, 48), 0), view_211, out=buf644)
        del view_211
        buf645 = reinterpret_tensor(buf639, (25088, 24), (24, 1), 0); del buf639  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf643, (25088, 48), (48, 1), 0), permute_655, out=buf645)
        del permute_655
        buf652 = reinterpret_tensor(buf593, (8, 197, 384), (75648, 384, 1), 0); del buf593  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_select_backward_slice_backward_31.run(buf600, buf652, 605184, grid=grid(605184), stream=stream0)
        buf653 = reinterpret_tensor(buf565, (1576, 1536), (1536, 1), 0); del buf565  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf652, (1576, 384), (384, 1), 0), permute_657, out=buf653)
        del permute_657
        buf657 = reinterpret_tensor(buf653, (8, 197, 1536), (302592, 1536, 1), 0); del buf653  # reuse
        # Source Nodes: [x_90], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf657, addmm_34, 2420736, grid=grid(2420736), stream=stream0)
        del addmm_34
        buf658 = reinterpret_tensor(buf600, (1576, 384), (384, 1), 0); del buf600  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf657, (1576, 1536), (1536, 1), 0), permute_661, out=buf658)
        del permute_661
        buf668 = reinterpret_tensor(buf590, (8, 197, 384), (75648, 384, 1), 0); del buf590  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf658, primals_146, mul_89, buf652, div_60, buf668, 1576, 384, grid=grid(1576), stream=stream0)
        del div_60
        del primals_146
        buf669 = buf588; del buf588  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf668, (1576, 384), (384, 1), 0), permute_665, out=buf669)
        del permute_665
        buf673 = reinterpret_tensor(buf576, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf576  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf669, buf673, 605184, grid=grid(605184), stream=stream0)
        buf674 = reinterpret_tensor(buf669, (48, 197, 64), (12608, 64, 1), 0); del buf669  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_670, reinterpret_tensor(buf673, (48, 197, 64), (12608, 64, 1), 0), out=buf674)
        del permute_670
        buf680 = buf566; del buf566  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf674, buf680, 605184, grid=grid(605184), stream=stream0)
        buf682 = reinterpret_tensor(buf674, (1576, 384), (384, 1), 0); del buf674  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf680, permute_678, out=buf682)
        del permute_678
        buf675 = reinterpret_tensor(buf585, (48, 197, 197), (38809, 197, 1), 0); del buf585  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf673, (48, 197, 64), (12608, 64, 1), 0), permute_671, out=buf675)
        del permute_671
        buf677 = reinterpret_tensor(buf583, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf583  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_12.run(buf675, alias_38, buf677, 9456, 197, grid=grid(9456), stream=stream0)
        del alias_38
        buf678 = reinterpret_tensor(buf673, (48, 64, 197), (12608, 197, 1), 0); del buf673  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_672, reinterpret_tensor(buf677, (48, 197, 197), (38809, 197, 1), 0), out=buf678)
        del permute_672
        buf679 = reinterpret_tensor(buf560, (48, 197, 64), (12608, 64, 1), 0); del buf560  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf677, (48, 197, 197), (38809, 197, 1), 0), permute_673, out=buf679)
        del permute_673
        buf683 = buf591; del buf591  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf679, buf678, buf683, 18912, 64, grid=grid(18912, 64), stream=stream0)
        buf685 = reinterpret_tensor(buf679, (1576, 384), (384, 1), 0); del buf679  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf683, (1576, 768), (768, 1), 0), permute_683, out=buf685)
        del permute_683
        buf692 = reinterpret_tensor(buf678, (8, 197, 384), (75648, 384, 1), 0); del buf678  # reuse
        # Source Nodes: [l__mod___blocks_4_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33.run(buf682, buf685, primals_140, cat_5, getitem_69, rsqrt_25, buf668, buf692, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_140
        buf693 = reinterpret_tensor(buf638, (1568, 384), (384, 1), 0); del buf638  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_17.run(buf692, buf693, 602112, grid=grid(602112), stream=stream0)
        buf694 = reinterpret_tensor(buf640, (1568, 384), (384, 1), 0); del buf640  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf693, permute_685, out=buf694)
        del permute_685
        buf704 = buf628; del buf628  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_34.run(buf704, buf642, buf645, primals_152, mul_84, buf694, primals_136, div_59, 25088, 24, grid=grid(25088), stream=stream0)
        del div_59
        del primals_136
        del primals_152
        buf648 = reinterpret_tensor(buf631, (24, 196), (1, 24), 0); del buf631  # reuse
        buf650 = buf624; del buf624  # reuse
        buf700 = buf558; del buf558  # reuse
        buf702 = buf556; del buf556  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_35.run(buf642, buf645, mul_84, buf694, buf648, buf650, buf700, buf702, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_84
        buf649 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf648, buf649, 24, 196, grid=grid(24), stream=stream0)
        buf651 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf650, buf651, 24, 196, grid=grid(24), stream=stream0)
        buf654 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf652, (384, 1576), (1, 384), 0), view_209, out=buf654)
        del view_209
        buf655 = buf604; del buf604  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf652, buf655, 4992, 122, grid=grid(4992), stream=stream0)
        buf656 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf655, buf656, 384, 13, grid=grid(384), stream=stream0)
        buf659 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf657, (1536, 1576), (1, 1536), 0), view_207, out=buf659)
        del view_207
        buf660 = buf568; del buf568  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf657, buf660, 19968, 122, grid=grid(19968), stream=stream0)
        buf661 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf660, buf661, 1536, 13, grid=grid(1536), stream=stream0)
        buf664 = reinterpret_tensor(buf655, (384, 13), (1, 384), 0); del buf655  # reuse
        buf666 = buf596; del buf596  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf658, mul_89, buf664, buf666, 4992, 122, grid=grid(4992), stream=stream0)
        del mul_89
        buf665 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf664, buf665, 384, 13, grid=grid(384), stream=stream0)
        buf667 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf666, buf667, 384, 13, grid=grid(384), stream=stream0)
        buf670 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf668, (384, 1576), (1, 384), 0), view_205, out=buf670)
        del view_205
        buf671 = reinterpret_tensor(buf666, (1, 384, 13), (4992, 1, 384), 0); del buf666  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf668, buf671, 4992, 122, grid=grid(4992), stream=stream0)
        buf672 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf671, buf672, 384, 13, grid=grid(384), stream=stream0)
        buf681 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf680, (384, 1576), (1, 384), 0), view_192, out=buf681)
        buf684 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf683, (768, 1576), (1, 768), 0), view_192, out=buf684)
        del view_192
        buf688 = reinterpret_tensor(buf671, (384, 13), (1, 384), 0); del buf671  # reuse
        buf690 = buf664; del buf664  # reuse
        # Source Nodes: [l__mod___blocks_4_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf682, buf685, cat_5, getitem_69, rsqrt_25, buf688, buf690, 4992, 122, grid=grid(4992), stream=stream0)
        del cat_5
        del getitem_69
        del rsqrt_25
        buf689 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_4_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf688, buf689, 384, 13, grid=grid(384), stream=stream0)
        buf691 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf690, buf691, 384, 13, grid=grid(384), stream=stream0)
        buf695 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf693, (384, 1568), (1, 384), 0), view_190, out=buf695)
        del view_190
        buf696 = reinterpret_tensor(buf690, (1, 384, 13), (4992, 1, 384), 0); del buf690  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf693, buf696, 4992, 121, grid=grid(4992), stream=stream0)
        buf697 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf696, buf697, 384, 13, grid=grid(384), stream=stream0)
        buf701 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf700, buf701, 24, 196, grid=grid(24), stream=stream0)
        buf703 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf702, buf703, 24, 196, grid=grid(24), stream=stream0)
        buf705 = reinterpret_tensor(buf617, (25088, 96), (96, 1), 0); del buf617  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf704, (25088, 24), (24, 1), 0), permute_689, out=buf705)
        del permute_689
        buf706 = empty((24, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf704, (24, 25088), (1, 24), 0), view_187, out=buf706)
        del view_187
        buf707 = reinterpret_tensor(buf702, (1, 24, 196), (4704, 1, 24), 0); del buf702  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf704, buf707, 4704, 128, grid=grid(4704), stream=stream0)
        buf708 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf707, buf708, 24, 196, grid=grid(24), stream=stream0)
        buf709 = reinterpret_tensor(buf705, (1568, 16, 96), (1536, 96, 1), 0); del buf705  # reuse
        # Source Nodes: [x_81], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf709, addmm_30, 2408448, grid=grid(2408448), stream=stream0)
        del addmm_30
        buf710 = reinterpret_tensor(buf693, (25088, 24), (24, 1), 0); del buf693  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf709, (25088, 96), (96, 1), 0), permute_693, out=buf710)
        del permute_693
        buf711 = empty((96, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf709, (96, 25088), (1, 96), 0), view_185, out=buf711)
        del view_185
        buf712 = buf620; del buf620  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf709, buf712, 18816, 128, grid=grid(18816), stream=stream0)
        buf713 = empty((1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf712, buf713, 96, 196, grid=grid(96), stream=stream0)
        buf720 = buf704; del buf704  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_26.run(buf720, buf710, primals_130, mul_79, div_63, 25088, 24, grid=grid(25088), stream=stream0)
        del div_63
        del primals_130
        buf716 = reinterpret_tensor(buf707, (24, 196), (1, 24), 0); del buf707  # reuse
        buf718 = buf700; del buf700  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_20.run(buf710, mul_79, buf716, buf718, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_79
        buf717 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf716, buf717, 24, 196, grid=grid(24), stream=stream0)
        buf719 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf718, buf719, 24, 196, grid=grid(24), stream=stream0)
        buf721 = buf710; del buf710  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf720, (25088, 24), (24, 1), 0), permute_697, out=buf721)
        del permute_697
        buf722 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf720, (24, 25088), (1, 24), 0), view_183, out=buf722)
        del view_183
        buf723 = reinterpret_tensor(buf718, (1, 24, 196), (4704, 1, 24), 0); del buf718  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf720, buf723, 4704, 128, grid=grid(4704), stream=stream0)
        buf724 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf723, buf724, 24, 196, grid=grid(24), stream=stream0)
        buf725 = reinterpret_tensor(buf694, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf694  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf721, buf725, 602112, grid=grid(602112), stream=stream0)
        buf726 = reinterpret_tensor(buf721, (6272, 16, 6), (96, 6, 1), 0); del buf721  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_702, reinterpret_tensor(buf725, (6272, 16, 6), (96, 6, 1), 0), out=buf726)
        del permute_702
        buf727 = reinterpret_tensor(buf637, (6272, 16, 16), (256, 16, 1), 0); del buf637  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf725, (6272, 16, 6), (96, 6, 1), 0), permute_703, out=buf727)
        del permute_703
        buf729 = reinterpret_tensor(buf635, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf635  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_28.run(buf727, alias_39, buf729, 100352, 16, grid=grid(100352), stream=stream0)
        del alias_39
        buf730 = reinterpret_tensor(buf725, (6272, 6, 16), (96, 16, 1), 0); del buf725  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_704, reinterpret_tensor(buf729, (6272, 16, 16), (256, 16, 1), 0), out=buf730)
        del permute_704
        buf731 = reinterpret_tensor(buf645, (6272, 16, 6), (96, 6, 1), 0); del buf645  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf729, (6272, 16, 16), (256, 16, 1), 0), permute_705, out=buf731)
        del permute_705
        buf732 = buf642; del buf642  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_29.run(buf726, buf732, 602112, grid=grid(602112), stream=stream0)
        buf733 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf732, (24, 25088), (1, 24), 0), view_170, out=buf733)
        buf734 = reinterpret_tensor(buf726, (25088, 24), (24, 1), 0); del buf726  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf732, permute_710, out=buf734)
        del permute_710
        buf735 = buf643; del buf643  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf731, buf730, buf735, 200704, 6, grid=grid(200704, 6), stream=stream0)
        buf736 = empty((48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf735, (48, 25088), (1, 48), 0), view_170, out=buf736)
        del view_170
        buf737 = reinterpret_tensor(buf731, (25088, 24), (24, 1), 0); del buf731  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf735, (25088, 48), (48, 1), 0), permute_715, out=buf737)
        del permute_715
        buf744 = reinterpret_tensor(buf685, (8, 197, 384), (75648, 384, 1), 0); del buf685  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_select_backward_slice_backward_31.run(buf692, buf744, 605184, grid=grid(605184), stream=stream0)
        buf745 = reinterpret_tensor(buf657, (1576, 1536), (1536, 1), 0); del buf657  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf744, (1576, 384), (384, 1), 0), permute_717, out=buf745)
        del permute_717
        buf749 = reinterpret_tensor(buf745, (8, 197, 1536), (302592, 1536, 1), 0); del buf745  # reuse
        # Source Nodes: [x_72], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf749, addmm_27, 2420736, grid=grid(2420736), stream=stream0)
        del addmm_27
        buf750 = reinterpret_tensor(buf692, (1576, 384), (384, 1), 0); del buf692  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf749, (1576, 1536), (1536, 1), 0), permute_721, out=buf750)
        del permute_721
        buf760 = reinterpret_tensor(buf682, (8, 197, 384), (75648, 384, 1), 0); del buf682  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf750, primals_118, mul_71, buf744, div_65, buf760, 1576, 384, grid=grid(1576), stream=stream0)
        del div_65
        del primals_118
        buf761 = buf680; del buf680  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf760, (1576, 384), (384, 1), 0), permute_725, out=buf761)
        del permute_725
        buf765 = reinterpret_tensor(buf668, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf668  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf761, buf765, 605184, grid=grid(605184), stream=stream0)
        buf766 = reinterpret_tensor(buf761, (48, 197, 64), (12608, 64, 1), 0); del buf761  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_730, reinterpret_tensor(buf765, (48, 197, 64), (12608, 64, 1), 0), out=buf766)
        del permute_730
        buf772 = buf658; del buf658  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf766, buf772, 605184, grid=grid(605184), stream=stream0)
        buf774 = reinterpret_tensor(buf766, (1576, 384), (384, 1), 0); del buf766  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf772, permute_738, out=buf774)
        del permute_738
        buf767 = reinterpret_tensor(buf677, (48, 197, 197), (38809, 197, 1), 0); del buf677  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf765, (48, 197, 64), (12608, 64, 1), 0), permute_731, out=buf767)
        del permute_731
        buf769 = reinterpret_tensor(buf675, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf675  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_12.run(buf767, alias_40, buf769, 9456, 197, grid=grid(9456), stream=stream0)
        del alias_40
        buf770 = reinterpret_tensor(buf765, (48, 64, 197), (12608, 197, 1), 0); del buf765  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_732, reinterpret_tensor(buf769, (48, 197, 197), (38809, 197, 1), 0), out=buf770)
        del permute_732
        buf771 = reinterpret_tensor(buf652, (48, 197, 64), (12608, 64, 1), 0); del buf652  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf769, (48, 197, 197), (38809, 197, 1), 0), permute_733, out=buf771)
        del permute_733
        buf775 = buf683; del buf683  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf771, buf770, buf775, 18912, 64, grid=grid(18912, 64), stream=stream0)
        buf777 = reinterpret_tensor(buf771, (1576, 384), (384, 1), 0); del buf771  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf775, (1576, 768), (768, 1), 0), permute_743, out=buf777)
        del permute_743
        buf784 = reinterpret_tensor(buf770, (8, 197, 384), (75648, 384, 1), 0); del buf770  # reuse
        # Source Nodes: [l__mod___blocks_3_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33.run(buf774, buf777, primals_112, cat_4, getitem_55, rsqrt_20, buf760, buf784, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_112
        buf785 = reinterpret_tensor(buf730, (1568, 384), (384, 1), 0); del buf730  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_17.run(buf784, buf785, 602112, grid=grid(602112), stream=stream0)
        buf786 = reinterpret_tensor(buf732, (1568, 384), (384, 1), 0); del buf732  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf785, permute_745, out=buf786)
        del permute_745
        buf796 = buf720; del buf720  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_34.run(buf796, buf734, buf737, primals_124, mul_66, buf786, primals_108, div_64, 25088, 24, grid=grid(25088), stream=stream0)
        del div_64
        del primals_108
        del primals_124
        buf740 = reinterpret_tensor(buf723, (24, 196), (1, 24), 0); del buf723  # reuse
        buf742 = buf716; del buf716  # reuse
        buf792 = buf650; del buf650  # reuse
        buf794 = buf648; del buf648  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_35.run(buf734, buf737, mul_66, buf786, buf740, buf742, buf792, buf794, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_66
        buf741 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf740, buf741, 24, 196, grid=grid(24), stream=stream0)
        buf743 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf742, buf743, 24, 196, grid=grid(24), stream=stream0)
        buf746 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf744, (384, 1576), (1, 384), 0), view_168, out=buf746)
        del view_168
        buf747 = buf696; del buf696  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf744, buf747, 4992, 122, grid=grid(4992), stream=stream0)
        buf748 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf747, buf748, 384, 13, grid=grid(384), stream=stream0)
        buf751 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf749, (1536, 1576), (1, 1536), 0), view_166, out=buf751)
        del view_166
        buf752 = buf660; del buf660  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf749, buf752, 19968, 122, grid=grid(19968), stream=stream0)
        buf753 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf752, buf753, 1536, 13, grid=grid(1536), stream=stream0)
        buf756 = reinterpret_tensor(buf747, (384, 13), (1, 384), 0); del buf747  # reuse
        buf758 = buf688; del buf688  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf750, mul_71, buf756, buf758, 4992, 122, grid=grid(4992), stream=stream0)
        del mul_71
        buf757 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf756, buf757, 384, 13, grid=grid(384), stream=stream0)
        buf759 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf758, buf759, 384, 13, grid=grid(384), stream=stream0)
        buf762 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf760, (384, 1576), (1, 384), 0), view_164, out=buf762)
        del view_164
        buf763 = reinterpret_tensor(buf758, (1, 384, 13), (4992, 1, 384), 0); del buf758  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf760, buf763, 4992, 122, grid=grid(4992), stream=stream0)
        buf764 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf763, buf764, 384, 13, grid=grid(384), stream=stream0)
        buf773 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf772, (384, 1576), (1, 384), 0), view_151, out=buf773)
        buf776 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf775, (768, 1576), (1, 768), 0), view_151, out=buf776)
        del view_151
        buf780 = reinterpret_tensor(buf763, (384, 13), (1, 384), 0); del buf763  # reuse
        buf782 = buf756; del buf756  # reuse
        # Source Nodes: [l__mod___blocks_3_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf774, buf777, cat_4, getitem_55, rsqrt_20, buf780, buf782, 4992, 122, grid=grid(4992), stream=stream0)
        del cat_4
        del getitem_55
        del rsqrt_20
        buf781 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf780, buf781, 384, 13, grid=grid(384), stream=stream0)
        buf783 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf782, buf783, 384, 13, grid=grid(384), stream=stream0)
        buf787 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf785, (384, 1568), (1, 384), 0), view_149, out=buf787)
        del view_149
        buf788 = reinterpret_tensor(buf782, (1, 384, 13), (4992, 1, 384), 0); del buf782  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf785, buf788, 4992, 121, grid=grid(4992), stream=stream0)
        buf789 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf788, buf789, 384, 13, grid=grid(384), stream=stream0)
        buf793 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf792, buf793, 24, 196, grid=grid(24), stream=stream0)
        buf795 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf794, buf795, 24, 196, grid=grid(24), stream=stream0)
        buf797 = reinterpret_tensor(buf709, (25088, 96), (96, 1), 0); del buf709  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf796, (25088, 24), (24, 1), 0), permute_749, out=buf797)
        del permute_749
        buf798 = empty((24, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf796, (24, 25088), (1, 24), 0), view_146, out=buf798)
        del view_146
        buf799 = reinterpret_tensor(buf794, (1, 24, 196), (4704, 1, 24), 0); del buf794  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf796, buf799, 4704, 128, grid=grid(4704), stream=stream0)
        buf800 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf799, buf800, 24, 196, grid=grid(24), stream=stream0)
        buf801 = reinterpret_tensor(buf797, (1568, 16, 96), (1536, 96, 1), 0); del buf797  # reuse
        # Source Nodes: [x_63], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf801, addmm_23, 2408448, grid=grid(2408448), stream=stream0)
        del addmm_23
        buf802 = reinterpret_tensor(buf785, (25088, 24), (24, 1), 0); del buf785  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf801, (25088, 96), (96, 1), 0), permute_753, out=buf802)
        del permute_753
        buf803 = empty((96, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf801, (96, 25088), (1, 96), 0), view_144, out=buf803)
        del view_144
        buf804 = buf712; del buf712  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf801, buf804, 18816, 128, grid=grid(18816), stream=stream0)
        buf805 = empty((1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf804, buf805, 96, 196, grid=grid(96), stream=stream0)
        buf812 = buf796; del buf796  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_26.run(buf812, buf802, primals_102, mul_61, div_68, 25088, 24, grid=grid(25088), stream=stream0)
        del div_68
        del primals_102
        buf808 = reinterpret_tensor(buf799, (24, 196), (1, 24), 0); del buf799  # reuse
        buf810 = buf792; del buf792  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_20.run(buf802, mul_61, buf808, buf810, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_61
        buf809 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf808, buf809, 24, 196, grid=grid(24), stream=stream0)
        buf811 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf810, buf811, 24, 196, grid=grid(24), stream=stream0)
        buf813 = buf802; del buf802  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf812, (25088, 24), (24, 1), 0), permute_757, out=buf813)
        del permute_757
        buf814 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf812, (24, 25088), (1, 24), 0), view_142, out=buf814)
        del view_142
        buf815 = reinterpret_tensor(buf810, (1, 24, 196), (4704, 1, 24), 0); del buf810  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf812, buf815, 4704, 128, grid=grid(4704), stream=stream0)
        buf816 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf815, buf816, 24, 196, grid=grid(24), stream=stream0)
        buf817 = reinterpret_tensor(buf786, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf786  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf813, buf817, 602112, grid=grid(602112), stream=stream0)
        buf818 = reinterpret_tensor(buf813, (6272, 16, 6), (96, 6, 1), 0); del buf813  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_762, reinterpret_tensor(buf817, (6272, 16, 6), (96, 6, 1), 0), out=buf818)
        del permute_762
        buf819 = reinterpret_tensor(buf729, (6272, 16, 16), (256, 16, 1), 0); del buf729  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf817, (6272, 16, 6), (96, 6, 1), 0), permute_763, out=buf819)
        del permute_763
        buf821 = reinterpret_tensor(buf727, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf727  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_28.run(buf819, alias_41, buf821, 100352, 16, grid=grid(100352), stream=stream0)
        del alias_41
        buf822 = reinterpret_tensor(buf817, (6272, 6, 16), (96, 16, 1), 0); del buf817  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_764, reinterpret_tensor(buf821, (6272, 16, 16), (256, 16, 1), 0), out=buf822)
        del permute_764
        buf823 = reinterpret_tensor(buf737, (6272, 16, 6), (96, 6, 1), 0); del buf737  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf821, (6272, 16, 16), (256, 16, 1), 0), permute_765, out=buf823)
        del permute_765
        buf824 = buf734; del buf734  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_29.run(buf818, buf824, 602112, grid=grid(602112), stream=stream0)
        buf825 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf824, (24, 25088), (1, 24), 0), view_129, out=buf825)
        buf826 = reinterpret_tensor(buf818, (25088, 24), (24, 1), 0); del buf818  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf824, permute_770, out=buf826)
        del permute_770
        buf827 = buf735; del buf735  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf823, buf822, buf827, 200704, 6, grid=grid(200704, 6), stream=stream0)
        buf828 = empty((48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf827, (48, 25088), (1, 48), 0), view_129, out=buf828)
        del view_129
        buf829 = reinterpret_tensor(buf823, (25088, 24), (24, 1), 0); del buf823  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf827, (25088, 48), (48, 1), 0), permute_775, out=buf829)
        del permute_775
        buf836 = reinterpret_tensor(buf777, (8, 197, 384), (75648, 384, 1), 0); del buf777  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_select_backward_slice_backward_31.run(buf784, buf836, 605184, grid=grid(605184), stream=stream0)
        buf837 = reinterpret_tensor(buf749, (1576, 1536), (1536, 1), 0); del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf836, (1576, 384), (384, 1), 0), permute_777, out=buf837)
        del permute_777
        buf841 = reinterpret_tensor(buf837, (8, 197, 1536), (302592, 1536, 1), 0); del buf837  # reuse
        # Source Nodes: [x_54], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf841, addmm_20, 2420736, grid=grid(2420736), stream=stream0)
        del addmm_20
        buf842 = reinterpret_tensor(buf784, (1576, 384), (384, 1), 0); del buf784  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf841, (1576, 1536), (1536, 1), 0), permute_781, out=buf842)
        del permute_781
        buf852 = reinterpret_tensor(buf774, (8, 197, 384), (75648, 384, 1), 0); del buf774  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf842, primals_90, mul_53, buf836, div_70, buf852, 1576, 384, grid=grid(1576), stream=stream0)
        del div_70
        del primals_90
        buf853 = buf772; del buf772  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf852, (1576, 384), (384, 1), 0), permute_785, out=buf853)
        del permute_785
        buf857 = reinterpret_tensor(buf760, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf760  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf853, buf857, 605184, grid=grid(605184), stream=stream0)
        buf858 = reinterpret_tensor(buf853, (48, 197, 64), (12608, 64, 1), 0); del buf853  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_790, reinterpret_tensor(buf857, (48, 197, 64), (12608, 64, 1), 0), out=buf858)
        del permute_790
        buf864 = buf750; del buf750  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf858, buf864, 605184, grid=grid(605184), stream=stream0)
        buf866 = reinterpret_tensor(buf858, (1576, 384), (384, 1), 0); del buf858  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf864, permute_798, out=buf866)
        del permute_798
        buf859 = reinterpret_tensor(buf769, (48, 197, 197), (38809, 197, 1), 0); del buf769  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf857, (48, 197, 64), (12608, 64, 1), 0), permute_791, out=buf859)
        del permute_791
        buf861 = reinterpret_tensor(buf767, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf767  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_12.run(buf859, alias_42, buf861, 9456, 197, grid=grid(9456), stream=stream0)
        del alias_42
        buf862 = reinterpret_tensor(buf857, (48, 64, 197), (12608, 197, 1), 0); del buf857  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_792, reinterpret_tensor(buf861, (48, 197, 197), (38809, 197, 1), 0), out=buf862)
        del permute_792
        buf863 = reinterpret_tensor(buf744, (48, 197, 64), (12608, 64, 1), 0); del buf744  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf861, (48, 197, 197), (38809, 197, 1), 0), permute_793, out=buf863)
        del permute_793
        buf867 = buf775; del buf775  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf863, buf862, buf867, 18912, 64, grid=grid(18912, 64), stream=stream0)
        buf869 = reinterpret_tensor(buf863, (1576, 384), (384, 1), 0); del buf863  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf867, (1576, 768), (768, 1), 0), permute_803, out=buf869)
        del permute_803
        buf876 = reinterpret_tensor(buf862, (8, 197, 384), (75648, 384, 1), 0); del buf862  # reuse
        # Source Nodes: [l__mod___blocks_2_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33.run(buf866, buf869, primals_84, cat_3, getitem_41, rsqrt_15, buf852, buf876, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_84
        buf877 = reinterpret_tensor(buf822, (1568, 384), (384, 1), 0); del buf822  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_17.run(buf876, buf877, 602112, grid=grid(602112), stream=stream0)
        buf878 = reinterpret_tensor(buf824, (1568, 384), (384, 1), 0); del buf824  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf877, permute_805, out=buf878)
        del permute_805
        buf888 = buf812; del buf812  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_34.run(buf888, buf826, buf829, primals_96, mul_48, buf878, primals_80, div_69, 25088, 24, grid=grid(25088), stream=stream0)
        del div_69
        del primals_80
        del primals_96
        buf832 = reinterpret_tensor(buf815, (24, 196), (1, 24), 0); del buf815  # reuse
        buf834 = buf808; del buf808  # reuse
        buf884 = buf742; del buf742  # reuse
        buf886 = buf740; del buf740  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_35.run(buf826, buf829, mul_48, buf878, buf832, buf834, buf884, buf886, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_48
        buf833 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf832, buf833, 24, 196, grid=grid(24), stream=stream0)
        buf835 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf834, buf835, 24, 196, grid=grid(24), stream=stream0)
        buf838 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf836, (384, 1576), (1, 384), 0), view_127, out=buf838)
        del view_127
        buf839 = buf788; del buf788  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf836, buf839, 4992, 122, grid=grid(4992), stream=stream0)
        buf840 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf839, buf840, 384, 13, grid=grid(384), stream=stream0)
        buf843 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf841, (1536, 1576), (1, 1536), 0), view_125, out=buf843)
        del view_125
        buf844 = buf752; del buf752  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf841, buf844, 19968, 122, grid=grid(19968), stream=stream0)
        buf845 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf844, buf845, 1536, 13, grid=grid(1536), stream=stream0)
        buf848 = reinterpret_tensor(buf839, (384, 13), (1, 384), 0); del buf839  # reuse
        buf850 = buf780; del buf780  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf842, mul_53, buf848, buf850, 4992, 122, grid=grid(4992), stream=stream0)
        del mul_53
        buf849 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf848, buf849, 384, 13, grid=grid(384), stream=stream0)
        buf851 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf850, buf851, 384, 13, grid=grid(384), stream=stream0)
        buf854 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf852, (384, 1576), (1, 384), 0), view_123, out=buf854)
        del view_123
        buf855 = reinterpret_tensor(buf850, (1, 384, 13), (4992, 1, 384), 0); del buf850  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf852, buf855, 4992, 122, grid=grid(4992), stream=stream0)
        buf856 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf855, buf856, 384, 13, grid=grid(384), stream=stream0)
        buf865 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf864, (384, 1576), (1, 384), 0), view_110, out=buf865)
        buf868 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf867, (768, 1576), (1, 768), 0), view_110, out=buf868)
        del view_110
        buf872 = reinterpret_tensor(buf855, (384, 13), (1, 384), 0); del buf855  # reuse
        buf874 = buf848; del buf848  # reuse
        # Source Nodes: [l__mod___blocks_2_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf866, buf869, cat_3, getitem_41, rsqrt_15, buf872, buf874, 4992, 122, grid=grid(4992), stream=stream0)
        del cat_3
        del getitem_41
        del rsqrt_15
        buf873 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf872, buf873, 384, 13, grid=grid(384), stream=stream0)
        buf875 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf874, buf875, 384, 13, grid=grid(384), stream=stream0)
        buf879 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf877, (384, 1568), (1, 384), 0), view_108, out=buf879)
        del view_108
        buf880 = reinterpret_tensor(buf874, (1, 384, 13), (4992, 1, 384), 0); del buf874  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf877, buf880, 4992, 121, grid=grid(4992), stream=stream0)
        buf881 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf880, buf881, 384, 13, grid=grid(384), stream=stream0)
        buf885 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf884, buf885, 24, 196, grid=grid(24), stream=stream0)
        buf887 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf886, buf887, 24, 196, grid=grid(24), stream=stream0)
        buf889 = reinterpret_tensor(buf801, (25088, 96), (96, 1), 0); del buf801  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf888, (25088, 24), (24, 1), 0), permute_809, out=buf889)
        del permute_809
        buf890 = empty((24, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf888, (24, 25088), (1, 24), 0), view_105, out=buf890)
        del view_105
        buf891 = reinterpret_tensor(buf886, (1, 24, 196), (4704, 1, 24), 0); del buf886  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf888, buf891, 4704, 128, grid=grid(4704), stream=stream0)
        buf892 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf891, buf892, 24, 196, grid=grid(24), stream=stream0)
        buf893 = reinterpret_tensor(buf889, (1568, 16, 96), (1536, 96, 1), 0); del buf889  # reuse
        # Source Nodes: [x_45], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf893, addmm_16, 2408448, grid=grid(2408448), stream=stream0)
        del addmm_16
        buf894 = reinterpret_tensor(buf877, (25088, 24), (24, 1), 0); del buf877  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf893, (25088, 96), (96, 1), 0), permute_813, out=buf894)
        del permute_813
        buf895 = empty((96, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf893, (96, 25088), (1, 96), 0), view_103, out=buf895)
        del view_103
        buf896 = buf804; del buf804  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf893, buf896, 18816, 128, grid=grid(18816), stream=stream0)
        buf897 = empty((1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf896, buf897, 96, 196, grid=grid(96), stream=stream0)
        buf904 = buf888; del buf888  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_26.run(buf904, buf894, primals_74, mul_43, div_73, 25088, 24, grid=grid(25088), stream=stream0)
        del div_73
        del primals_74
        buf900 = reinterpret_tensor(buf891, (24, 196), (1, 24), 0); del buf891  # reuse
        buf902 = buf884; del buf884  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_20.run(buf894, mul_43, buf900, buf902, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_43
        buf901 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf900, buf901, 24, 196, grid=grid(24), stream=stream0)
        buf903 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf902, buf903, 24, 196, grid=grid(24), stream=stream0)
        buf905 = buf894; del buf894  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf904, (25088, 24), (24, 1), 0), permute_817, out=buf905)
        del permute_817
        buf906 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf904, (24, 25088), (1, 24), 0), view_101, out=buf906)
        del view_101
        buf907 = reinterpret_tensor(buf902, (1, 24, 196), (4704, 1, 24), 0); del buf902  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf904, buf907, 4704, 128, grid=grid(4704), stream=stream0)
        buf908 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf907, buf908, 24, 196, grid=grid(24), stream=stream0)
        buf909 = reinterpret_tensor(buf878, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf878  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf905, buf909, 602112, grid=grid(602112), stream=stream0)
        buf910 = reinterpret_tensor(buf905, (6272, 16, 6), (96, 6, 1), 0); del buf905  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_822, reinterpret_tensor(buf909, (6272, 16, 6), (96, 6, 1), 0), out=buf910)
        del permute_822
        buf911 = reinterpret_tensor(buf821, (6272, 16, 16), (256, 16, 1), 0); del buf821  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf909, (6272, 16, 6), (96, 6, 1), 0), permute_823, out=buf911)
        del permute_823
        buf913 = reinterpret_tensor(buf819, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf819  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_28.run(buf911, alias_43, buf913, 100352, 16, grid=grid(100352), stream=stream0)
        del alias_43
        buf914 = reinterpret_tensor(buf909, (6272, 6, 16), (96, 16, 1), 0); del buf909  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_824, reinterpret_tensor(buf913, (6272, 16, 16), (256, 16, 1), 0), out=buf914)
        del permute_824
        buf915 = reinterpret_tensor(buf829, (6272, 16, 6), (96, 6, 1), 0); del buf829  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf913, (6272, 16, 16), (256, 16, 1), 0), permute_825, out=buf915)
        del permute_825
        buf916 = buf826; del buf826  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_29.run(buf910, buf916, 602112, grid=grid(602112), stream=stream0)
        buf917 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf916, (24, 25088), (1, 24), 0), view_88, out=buf917)
        buf918 = reinterpret_tensor(buf910, (25088, 24), (24, 1), 0); del buf910  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf916, permute_830, out=buf918)
        del permute_830
        buf919 = buf827; del buf827  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf915, buf914, buf919, 200704, 6, grid=grid(200704, 6), stream=stream0)
        buf920 = empty((48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf919, (48, 25088), (1, 48), 0), view_88, out=buf920)
        del view_88
        buf921 = reinterpret_tensor(buf915, (25088, 24), (24, 1), 0); del buf915  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf919, (25088, 48), (48, 1), 0), permute_835, out=buf921)
        del permute_835
        buf928 = reinterpret_tensor(buf869, (8, 197, 384), (75648, 384, 1), 0); del buf869  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_select_backward_slice_backward_31.run(buf876, buf928, 605184, grid=grid(605184), stream=stream0)
        buf929 = reinterpret_tensor(buf841, (1576, 1536), (1536, 1), 0); del buf841  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf928, (1576, 384), (384, 1), 0), permute_837, out=buf929)
        del permute_837
        buf933 = reinterpret_tensor(buf929, (8, 197, 1536), (302592, 1536, 1), 0); del buf929  # reuse
        # Source Nodes: [x_36], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf933, addmm_13, 2420736, grid=grid(2420736), stream=stream0)
        del addmm_13
        buf934 = reinterpret_tensor(buf876, (1576, 384), (384, 1), 0); del buf876  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf933, (1576, 1536), (1536, 1), 0), permute_841, out=buf934)
        del permute_841
        buf944 = reinterpret_tensor(buf866, (8, 197, 384), (75648, 384, 1), 0); del buf866  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf934, primals_62, mul_35, buf928, div_75, buf944, 1576, 384, grid=grid(1576), stream=stream0)
        del div_75
        del primals_62
        buf945 = buf864; del buf864  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf944, (1576, 384), (384, 1), 0), permute_845, out=buf945)
        del permute_845
        buf949 = reinterpret_tensor(buf852, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf852  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf945, buf949, 605184, grid=grid(605184), stream=stream0)
        buf950 = reinterpret_tensor(buf945, (48, 197, 64), (12608, 64, 1), 0); del buf945  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_850, reinterpret_tensor(buf949, (48, 197, 64), (12608, 64, 1), 0), out=buf950)
        del permute_850
        buf956 = buf842; del buf842  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf950, buf956, 605184, grid=grid(605184), stream=stream0)
        buf958 = reinterpret_tensor(buf950, (1576, 384), (384, 1), 0); del buf950  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf956, permute_858, out=buf958)
        del permute_858
        buf951 = reinterpret_tensor(buf861, (48, 197, 197), (38809, 197, 1), 0); del buf861  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf949, (48, 197, 64), (12608, 64, 1), 0), permute_851, out=buf951)
        del permute_851
        buf953 = reinterpret_tensor(buf859, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf859  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_12.run(buf951, alias_44, buf953, 9456, 197, grid=grid(9456), stream=stream0)
        del alias_44
        buf954 = reinterpret_tensor(buf949, (48, 64, 197), (12608, 197, 1), 0); del buf949  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_852, reinterpret_tensor(buf953, (48, 197, 197), (38809, 197, 1), 0), out=buf954)
        del permute_852
        buf955 = reinterpret_tensor(buf836, (48, 197, 64), (12608, 64, 1), 0); del buf836  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf953, (48, 197, 197), (38809, 197, 1), 0), permute_853, out=buf955)
        del permute_853
        buf959 = buf867; del buf867  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf955, buf954, buf959, 18912, 64, grid=grid(18912, 64), stream=stream0)
        buf961 = reinterpret_tensor(buf955, (1576, 384), (384, 1), 0); del buf955  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf959, (1576, 768), (768, 1), 0), permute_863, out=buf961)
        del permute_863
        buf968 = reinterpret_tensor(buf954, (8, 197, 384), (75648, 384, 1), 0); del buf954  # reuse
        # Source Nodes: [l__mod___blocks_1_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33.run(buf958, buf961, primals_56, cat_2, getitem_27, rsqrt_10, buf944, buf968, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_56
        buf969 = reinterpret_tensor(buf914, (1568, 384), (384, 1), 0); del buf914  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_17.run(buf968, buf969, 602112, grid=grid(602112), stream=stream0)
        buf970 = reinterpret_tensor(buf916, (1568, 384), (384, 1), 0); del buf916  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf969, permute_865, out=buf970)
        del permute_865
        buf980 = buf904; del buf904  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_34.run(buf980, buf918, buf921, primals_68, mul_30, buf970, primals_52, div_74, 25088, 24, grid=grid(25088), stream=stream0)
        del div_74
        del primals_52
        del primals_68
        buf924 = reinterpret_tensor(buf907, (24, 196), (1, 24), 0); del buf907  # reuse
        buf926 = buf900; del buf900  # reuse
        buf976 = buf834; del buf834  # reuse
        buf978 = buf832; del buf832  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_35.run(buf918, buf921, mul_30, buf970, buf924, buf926, buf976, buf978, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_30
        buf925 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf924, buf925, 24, 196, grid=grid(24), stream=stream0)
        buf927 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf926, buf927, 24, 196, grid=grid(24), stream=stream0)
        buf930 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf928, (384, 1576), (1, 384), 0), view_86, out=buf930)
        del view_86
        buf931 = buf880; del buf880  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf928, buf931, 4992, 122, grid=grid(4992), stream=stream0)
        buf932 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf931, buf932, 384, 13, grid=grid(384), stream=stream0)
        buf935 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf933, (1536, 1576), (1, 1536), 0), view_84, out=buf935)
        del view_84
        buf936 = buf844; del buf844  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf933, buf936, 19968, 122, grid=grid(19968), stream=stream0)
        buf937 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf936, buf937, 1536, 13, grid=grid(1536), stream=stream0)
        buf940 = reinterpret_tensor(buf931, (384, 13), (1, 384), 0); del buf931  # reuse
        buf942 = buf872; del buf872  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf934, mul_35, buf940, buf942, 4992, 122, grid=grid(4992), stream=stream0)
        del mul_35
        buf941 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf940, buf941, 384, 13, grid=grid(384), stream=stream0)
        buf943 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf942, buf943, 384, 13, grid=grid(384), stream=stream0)
        buf946 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf944, (384, 1576), (1, 384), 0), view_82, out=buf946)
        del view_82
        buf947 = reinterpret_tensor(buf942, (1, 384, 13), (4992, 1, 384), 0); del buf942  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf944, buf947, 4992, 122, grid=grid(4992), stream=stream0)
        buf948 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf947, buf948, 384, 13, grid=grid(384), stream=stream0)
        buf957 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf956, (384, 1576), (1, 384), 0), view_69, out=buf957)
        buf960 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf959, (768, 1576), (1, 768), 0), view_69, out=buf960)
        del view_69
        buf964 = reinterpret_tensor(buf947, (384, 13), (1, 384), 0); del buf947  # reuse
        buf966 = buf940; del buf940  # reuse
        # Source Nodes: [l__mod___blocks_1_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf958, buf961, cat_2, getitem_27, rsqrt_10, buf964, buf966, 4992, 122, grid=grid(4992), stream=stream0)
        del cat_2
        del getitem_27
        del rsqrt_10
        buf965 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf964, buf965, 384, 13, grid=grid(384), stream=stream0)
        buf967 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf966, buf967, 384, 13, grid=grid(384), stream=stream0)
        buf971 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf969, (384, 1568), (1, 384), 0), view_67, out=buf971)
        del view_67
        buf972 = reinterpret_tensor(buf966, (1, 384, 13), (4992, 1, 384), 0); del buf966  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf969, buf972, 4992, 121, grid=grid(4992), stream=stream0)
        buf973 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf972, buf973, 384, 13, grid=grid(384), stream=stream0)
        buf977 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf976, buf977, 24, 196, grid=grid(24), stream=stream0)
        buf979 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf978, buf979, 24, 196, grid=grid(24), stream=stream0)
        buf981 = reinterpret_tensor(buf893, (25088, 96), (96, 1), 0); del buf893  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf980, (25088, 24), (24, 1), 0), permute_869, out=buf981)
        del permute_869
        buf982 = empty((24, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf980, (24, 25088), (1, 24), 0), view_64, out=buf982)
        del view_64
        buf983 = reinterpret_tensor(buf978, (1, 24, 196), (4704, 1, 24), 0); del buf978  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf980, buf983, 4704, 128, grid=grid(4704), stream=stream0)
        buf984 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf983, buf984, 24, 196, grid=grid(24), stream=stream0)
        buf985 = reinterpret_tensor(buf981, (1568, 16, 96), (1536, 96, 1), 0); del buf981  # reuse
        # Source Nodes: [x_27], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf985, addmm_9, 2408448, grid=grid(2408448), stream=stream0)
        del addmm_9
        buf986 = reinterpret_tensor(buf969, (25088, 24), (24, 1), 0); del buf969  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf985, (25088, 96), (96, 1), 0), permute_873, out=buf986)
        del permute_873
        buf987 = empty((96, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf985, (96, 25088), (1, 96), 0), view_62, out=buf987)
        del view_62
        buf988 = buf896; del buf896  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf985, buf988, 18816, 128, grid=grid(18816), stream=stream0)
        buf989 = empty((1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf988, buf989, 96, 196, grid=grid(96), stream=stream0)
        buf996 = buf980; del buf980  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_26.run(buf996, buf986, primals_46, mul_25, div_78, 25088, 24, grid=grid(25088), stream=stream0)
        del div_78
        del primals_46
        buf992 = reinterpret_tensor(buf983, (24, 196), (1, 24), 0); del buf983  # reuse
        buf994 = buf976; del buf976  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_20.run(buf986, mul_25, buf992, buf994, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_25
        buf993 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf992, buf993, 24, 196, grid=grid(24), stream=stream0)
        buf995 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf994, buf995, 24, 196, grid=grid(24), stream=stream0)
        buf997 = buf986; del buf986  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf996, (25088, 24), (24, 1), 0), permute_877, out=buf997)
        del permute_877
        buf998 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf996, (24, 25088), (1, 24), 0), view_60, out=buf998)
        del view_60
        buf999 = reinterpret_tensor(buf994, (1, 24, 196), (4704, 1, 24), 0); del buf994  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf996, buf999, 4704, 128, grid=grid(4704), stream=stream0)
        buf1000 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf999, buf1000, 24, 196, grid=grid(24), stream=stream0)
        buf1001 = reinterpret_tensor(buf970, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf970  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf997, buf1001, 602112, grid=grid(602112), stream=stream0)
        buf1002 = reinterpret_tensor(buf997, (6272, 16, 6), (96, 6, 1), 0); del buf997  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_882, reinterpret_tensor(buf1001, (6272, 16, 6), (96, 6, 1), 0), out=buf1002)
        del permute_882
        buf1003 = reinterpret_tensor(buf913, (6272, 16, 16), (256, 16, 1), 0); del buf913  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1001, (6272, 16, 6), (96, 6, 1), 0), permute_883, out=buf1003)
        del permute_883
        buf1005 = reinterpret_tensor(buf911, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf911  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_28.run(buf1003, alias_45, buf1005, 100352, 16, grid=grid(100352), stream=stream0)
        del alias_45
        buf1006 = reinterpret_tensor(buf1001, (6272, 6, 16), (96, 16, 1), 0); del buf1001  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_884, reinterpret_tensor(buf1005, (6272, 16, 16), (256, 16, 1), 0), out=buf1006)
        del permute_884
        buf1007 = reinterpret_tensor(buf921, (6272, 16, 6), (96, 6, 1), 0); del buf921  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1005, (6272, 16, 16), (256, 16, 1), 0), permute_885, out=buf1007)
        del permute_885
        buf1008 = buf918; del buf918  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_29.run(buf1002, buf1008, 602112, grid=grid(602112), stream=stream0)
        buf1009 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1008, (24, 25088), (1, 24), 0), view_47, out=buf1009)
        buf1010 = reinterpret_tensor(buf1002, (25088, 24), (24, 1), 0); del buf1002  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1008, permute_890, out=buf1010)
        del permute_890
        buf1011 = buf919; del buf919  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf1007, buf1006, buf1011, 200704, 6, grid=grid(200704, 6), stream=stream0)
        buf1012 = empty((48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1011, (48, 25088), (1, 48), 0), view_47, out=buf1012)
        del view_47
        buf1013 = reinterpret_tensor(buf1007, (25088, 24), (24, 1), 0); del buf1007  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1011, (25088, 48), (48, 1), 0), permute_895, out=buf1013)
        del permute_895
        buf1020 = reinterpret_tensor(buf961, (8, 197, 384), (75648, 384, 1), 0); del buf961  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_select_backward_slice_backward_31.run(buf968, buf1020, 605184, grid=grid(605184), stream=stream0)
        buf1021 = reinterpret_tensor(buf933, (1576, 1536), (1536, 1), 0); del buf933  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1020, (1576, 384), (384, 1), 0), permute_897, out=buf1021)
        del permute_897
        buf1025 = reinterpret_tensor(buf1021, (8, 197, 1536), (302592, 1536, 1), 0); del buf1021  # reuse
        # Source Nodes: [x_18], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf1025, addmm_6, 2420736, grid=grid(2420736), stream=stream0)
        del addmm_6
        buf1026 = reinterpret_tensor(buf968, (1576, 384), (384, 1), 0); del buf968  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1025, (1576, 1536), (1536, 1), 0), permute_901, out=buf1026)
        del permute_901
        buf1036 = reinterpret_tensor(buf958, (8, 197, 384), (75648, 384, 1), 0); del buf958  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf1026, primals_34, mul_17, buf1020, div_80, buf1036, 1576, 384, grid=grid(1576), stream=stream0)
        del div_80
        del primals_34
        buf1037 = buf956; del buf956  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1036, (1576, 384), (384, 1), 0), permute_905, out=buf1037)
        del permute_905
        buf1041 = reinterpret_tensor(buf944, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf944  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf1037, buf1041, 605184, grid=grid(605184), stream=stream0)
        buf1042 = reinterpret_tensor(buf1037, (48, 197, 64), (12608, 64, 1), 0); del buf1037  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_910, reinterpret_tensor(buf1041, (48, 197, 64), (12608, 64, 1), 0), out=buf1042)
        del permute_910
        buf1048 = buf934; del buf934  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf1042, buf1048, 605184, grid=grid(605184), stream=stream0)
        buf1050 = reinterpret_tensor(buf1042, (1576, 384), (384, 1), 0); del buf1042  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1048, permute_918, out=buf1050)
        del permute_918
        buf1043 = reinterpret_tensor(buf953, (48, 197, 197), (38809, 197, 1), 0); del buf953  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1041, (48, 197, 64), (12608, 64, 1), 0), permute_911, out=buf1043)
        del permute_911
        buf1045 = reinterpret_tensor(buf951, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf951  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_12.run(buf1043, alias_46, buf1045, 9456, 197, grid=grid(9456), stream=stream0)
        del alias_46
        del buf1043
        buf1046 = reinterpret_tensor(buf1041, (48, 64, 197), (12608, 197, 1), 0); del buf1041  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_912, reinterpret_tensor(buf1045, (48, 197, 197), (38809, 197, 1), 0), out=buf1046)
        del permute_912
        buf1047 = reinterpret_tensor(buf928, (48, 197, 64), (12608, 64, 1), 0); del buf928  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1045, (48, 197, 197), (38809, 197, 1), 0), permute_913, out=buf1047)
        del buf1045
        del permute_913
        buf1051 = buf959; del buf959  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf1047, buf1046, buf1051, 18912, 64, grid=grid(18912, 64), stream=stream0)
        buf1053 = reinterpret_tensor(buf1047, (1576, 384), (384, 1), 0); del buf1047  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1051, (1576, 768), (768, 1), 0), permute_923, out=buf1053)
        del permute_923
        buf1060 = reinterpret_tensor(buf1046, (8, 197, 384), (75648, 384, 1), 0); del buf1046  # reuse
        # Source Nodes: [l__mod___blocks_0_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33.run(buf1050, buf1053, primals_28, cat_1, getitem_13, rsqrt_5, buf1036, buf1060, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_28
        buf1061 = reinterpret_tensor(buf1006, (1568, 384), (384, 1), 0); del buf1006  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_17.run(buf1060, buf1061, 602112, grid=grid(602112), stream=stream0)
        buf1062 = reinterpret_tensor(buf1008, (1568, 384), (384, 1), 0); del buf1008  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1061, permute_925, out=buf1062)
        del permute_925
        buf1072 = buf996; del buf996  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_34.run(buf1072, buf1010, buf1013, primals_40, mul_12, buf1062, primals_24, div_79, 25088, 24, grid=grid(25088), stream=stream0)
        del div_79
        del primals_24
        del primals_40
        buf1016 = reinterpret_tensor(buf999, (24, 196), (1, 24), 0); del buf999  # reuse
        buf1018 = buf992; del buf992  # reuse
        buf1068 = buf926; del buf926  # reuse
        buf1070 = buf924; del buf924  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_35.run(buf1010, buf1013, mul_12, buf1062, buf1016, buf1018, buf1068, buf1070, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_12
        buf1017 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf1016, buf1017, 24, 196, grid=grid(24), stream=stream0)
        del buf1016
        buf1019 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf1018, buf1019, 24, 196, grid=grid(24), stream=stream0)
        del buf1018
        buf1022 = empty((384, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1020, (384, 1576), (1, 384), 0), view_45, out=buf1022)
        del view_45
        buf1023 = buf972; del buf972  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf1020, buf1023, 4992, 122, grid=grid(4992), stream=stream0)
        del buf1020
        buf1024 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf1023, buf1024, 384, 13, grid=grid(384), stream=stream0)
        buf1027 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1025, (1536, 1576), (1, 1536), 0), view_43, out=buf1027)
        del view_43
        buf1028 = buf936; del buf936  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf1025, buf1028, 19968, 122, grid=grid(19968), stream=stream0)
        del buf1025
        buf1029 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf1028, buf1029, 1536, 13, grid=grid(1536), stream=stream0)
        del buf1028
        buf1032 = reinterpret_tensor(buf1023, (384, 13), (1, 384), 0); del buf1023  # reuse
        buf1034 = buf964; del buf964  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf1026, mul_17, buf1032, buf1034, 4992, 122, grid=grid(4992), stream=stream0)
        del buf1026
        del mul_17
        buf1033 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf1032, buf1033, 384, 13, grid=grid(384), stream=stream0)
        buf1035 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf1034, buf1035, 384, 13, grid=grid(384), stream=stream0)
        buf1038 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1036, (384, 1576), (1, 384), 0), view_41, out=buf1038)
        del view_41
        buf1039 = reinterpret_tensor(buf1034, (1, 384, 13), (4992, 1, 384), 0); del buf1034  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf1036, buf1039, 4992, 122, grid=grid(4992), stream=stream0)
        del buf1036
        buf1040 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf1039, buf1040, 384, 13, grid=grid(384), stream=stream0)
        buf1049 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1048, (384, 1576), (1, 384), 0), view_28, out=buf1049)
        del buf1048
        buf1052 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1051, (768, 1576), (1, 768), 0), view_28, out=buf1052)
        del buf1051
        del view_28
        buf1056 = reinterpret_tensor(buf1039, (384, 13), (1, 384), 0); del buf1039  # reuse
        buf1058 = buf1032; del buf1032  # reuse
        # Source Nodes: [l__mod___blocks_0_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf1050, buf1053, cat_1, getitem_13, rsqrt_5, buf1056, buf1058, 4992, 122, grid=grid(4992), stream=stream0)
        del buf1050
        del buf1053
        del cat_1
        del getitem_13
        del rsqrt_5
        buf1057 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_norm_out], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf1056, buf1057, 384, 13, grid=grid(384), stream=stream0)
        buf1059 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf1058, buf1059, 384, 13, grid=grid(384), stream=stream0)
        buf1063 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1061, (384, 1568), (1, 384), 0), view_26, out=buf1063)
        del view_26
        buf1064 = reinterpret_tensor(buf1058, (1, 384, 13), (4992, 1, 384), 0); del buf1058  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf1061, buf1064, 4992, 121, grid=grid(4992), stream=stream0)
        buf1065 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf1064, buf1065, 384, 13, grid=grid(384), stream=stream0)
        buf1069 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf1068, buf1069, 24, 196, grid=grid(24), stream=stream0)
        buf1071 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf1070, buf1071, 24, 196, grid=grid(24), stream=stream0)
        buf1073 = reinterpret_tensor(buf985, (25088, 96), (96, 1), 0); del buf985  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1072, (25088, 24), (24, 1), 0), permute_929, out=buf1073)
        del permute_929
        buf1074 = empty((24, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1072, (24, 25088), (1, 24), 0), view_23, out=buf1074)
        del view_23
        buf1075 = reinterpret_tensor(buf1070, (1, 24, 196), (4704, 1, 24), 0); del buf1070  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf1072, buf1075, 4704, 128, grid=grid(4704), stream=stream0)
        buf1076 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf1075, buf1076, 24, 196, grid=grid(24), stream=stream0)
        buf1077 = reinterpret_tensor(buf1073, (1568, 16, 96), (1536, 96, 1), 0); del buf1073  # reuse
        # Source Nodes: [x_9], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf1077, addmm_2, 2408448, grid=grid(2408448), stream=stream0)
        del addmm_2
        buf1078 = reinterpret_tensor(buf1061, (25088, 24), (24, 1), 0); del buf1061  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1077, (25088, 96), (96, 1), 0), permute_933, out=buf1078)
        del permute_933
        buf1079 = empty((96, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1077, (96, 25088), (1, 96), 0), view_21, out=buf1079)
        del view_21
        buf1080 = buf988; del buf988  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1077, buf1080, 18816, 128, grid=grid(18816), stream=stream0)
        del buf1077
        buf1081 = empty((1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf1080, buf1081, 96, 196, grid=grid(96), stream=stream0)
        del buf1080
        buf1088 = buf1072; del buf1072  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_26.run(buf1088, buf1078, primals_18, mul_7, div_83, 25088, 24, grid=grid(25088), stream=stream0)
        del div_83
        del primals_18
        buf1084 = reinterpret_tensor(buf1075, (24, 196), (1, 24), 0); del buf1075  # reuse
        buf1086 = buf1068; del buf1068  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_20.run(buf1078, mul_7, buf1084, buf1086, 4704, 128, grid=grid(4704), stream=stream0)
        del mul_7
        buf1085 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf1084, buf1085, 24, 196, grid=grid(24), stream=stream0)
        buf1087 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf1086, buf1087, 24, 196, grid=grid(24), stream=stream0)
        buf1089 = buf1078; del buf1078  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1088, (25088, 24), (24, 1), 0), permute_937, out=buf1089)
        del permute_937
        buf1090 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1088, (24, 25088), (1, 24), 0), view_19, out=buf1090)
        del view_19
        buf1091 = reinterpret_tensor(buf1086, (1, 24, 196), (4704, 1, 24), 0); del buf1086  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf1088, buf1091, 4704, 128, grid=grid(4704), stream=stream0)
        buf1092 = empty((1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_native_layer_norm_backward_21.run(buf1091, buf1092, 24, 196, grid=grid(24), stream=stream0)
        buf1093 = reinterpret_tensor(buf1062, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf1062  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf1089, buf1093, 602112, grid=grid(602112), stream=stream0)
        buf1094 = reinterpret_tensor(buf1089, (6272, 16, 6), (96, 6, 1), 0); del buf1089  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_942, reinterpret_tensor(buf1093, (6272, 16, 6), (96, 6, 1), 0), out=buf1094)
        del permute_942
        buf1095 = reinterpret_tensor(buf1005, (6272, 16, 16), (256, 16, 1), 0); del buf1005  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1093, (6272, 16, 6), (96, 6, 1), 0), permute_943, out=buf1095)
        del permute_943
        buf1097 = reinterpret_tensor(buf1003, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf1003  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_28.run(buf1095, alias_47, buf1097, 100352, 16, grid=grid(100352), stream=stream0)
        del alias_47
        del buf1095
        buf1098 = reinterpret_tensor(buf1093, (6272, 6, 16), (96, 16, 1), 0); del buf1093  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_944, reinterpret_tensor(buf1097, (6272, 16, 16), (256, 16, 1), 0), out=buf1098)
        del permute_944
        buf1099 = reinterpret_tensor(buf1013, (6272, 16, 6), (96, 6, 1), 0); del buf1013  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1097, (6272, 16, 16), (256, 16, 1), 0), permute_945, out=buf1099)
        del buf1097
        del permute_945
        buf1100 = buf1010; del buf1010  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_29.run(buf1094, buf1100, 602112, grid=grid(602112), stream=stream0)
        buf1101 = empty((24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1100, (24, 25088), (1, 24), 0), view_6, out=buf1101)
        buf1102 = reinterpret_tensor(buf1094, (25088, 24), (24, 1), 0); del buf1094  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1100, permute_950, out=buf1102)
        del permute_950
        buf1103 = buf1011; del buf1011  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf1099, buf1098, buf1103, 200704, 6, grid=grid(200704, 6), stream=stream0)
        buf1104 = empty((48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1103, (48, 25088), (1, 48), 0), view_6, out=buf1104)
        del view_6
        buf1105 = reinterpret_tensor(buf1099, (25088, 24), (24, 1), 0); del buf1099  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1103, (25088, 48), (48, 1), 0), permute_955, out=buf1105)
        del buf1103
        del permute_955
        buf1121 = reinterpret_tensor(buf1098, (8, 196, 384), (75264, 384, 1), 0); del buf1098  # reuse
        # Source Nodes: [patch_embed], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_37.run(buf1060, primals_10, addmm, getitem_3, rsqrt_1, buf1121, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_10
        buf1122 = reinterpret_tensor(buf1100, (1568, 384), (384, 1), 0); del buf1100  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1121, (1568, 384), (384, 1), 0), permute_957, out=buf1122)
        del permute_957
        buf1126 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf1127 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___norm1_proj], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_38.run(buf1122, primals_6, clone_2, getitem_1, rsqrt, buf1126, buf1127, 1568, 384, grid=grid(1568), stream=stream0)
        buf1112 = buf1088; del buf1088  # reuse
        buf1132 = buf1112; del buf1112  # reuse
        # Source Nodes: [l__mod___blocks_0_norm_in], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_39.run(buf1132, buf1102, buf1105, primals_12, clone_2, getitem_5, rsqrt_2, rsqrt, buf1122, primals_6, buf1126, getitem_1, buf1127, 25088, 24, grid=grid(25088), stream=stream0)
        del buf1126
        del buf1127
        del primals_12
        del primals_6
        buf1108 = reinterpret_tensor(buf1091, (24, 196), (1, 24), 0); del buf1091  # reuse
        buf1110 = buf1084; del buf1084  # reuse
        # Source Nodes: [l__mod___blocks_0_norm_in], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_40.run(buf1102, buf1105, clone_2, getitem_5, rsqrt_2, buf1108, buf1110, 4704, 128, grid=grid(4704), stream=stream0)
        del buf1102
        del buf1105
        del getitem_5
        del rsqrt_2
        buf1109 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_norm_in], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf1108, buf1109, 24, 196, grid=grid(24), stream=stream0)
        del buf1108
        buf1111 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf1110, buf1111, 24, 196, grid=grid(24), stream=stream0)
        del buf1110
        buf1113 = empty((1, 197, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward, aten.sum]
        triton_per_fused_add_select_backward_slice_backward_sum_41.run(buf1060, buf1113, 75648, 8, grid=grid(75648), stream=stream0)
        buf1114 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_42.run(buf1060, buf1114, 384, 8, grid=grid(384), stream=stream0)
        buf1117 = reinterpret_tensor(buf1064, (384, 13), (1, 384), 0); del buf1064  # reuse
        buf1119 = buf1056; del buf1056  # reuse
        # Source Nodes: [patch_embed], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_43.run(buf1060, addmm, getitem_3, rsqrt_1, buf1117, buf1119, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm
        del buf1060
        del getitem_3
        del rsqrt_1
        buf1118 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [patch_embed], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf1117, buf1118, 384, 13, grid=grid(384), stream=stream0)
        buf1120 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf1119, buf1120, 384, 13, grid=grid(384), stream=stream0)
        buf1123 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1121, (384, 1568), (1, 384), 0), view_4, out=buf1123)
        del view_4
        buf1124 = reinterpret_tensor(buf1119, (1, 384, 13), (4992, 1, 384), 0); del buf1119  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf1121, buf1124, 4992, 121, grid=grid(4992), stream=stream0)
        buf1125 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf1124, buf1125, 384, 13, grid=grid(384), stream=stream0)
        buf1128 = reinterpret_tensor(buf1124, (384, 13), (1, 384), 0); del buf1124  # reuse
        buf1130 = buf1117; del buf1117  # reuse
        # Source Nodes: [l__mod___norm1_proj], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_44.run(buf1122, clone_2, getitem_1, rsqrt, buf1128, buf1130, 4992, 121, grid=grid(4992), stream=stream0)
        del clone_2
        del getitem_1
        del rsqrt
        buf1129 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___norm1_proj], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf1128, buf1129, 384, 13, grid=grid(384), stream=stream0)
        del buf1128
        buf1131 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf1130, buf1131, 384, 13, grid=grid(384), stream=stream0)
        buf1133 = reinterpret_tensor(buf1130, (1, 24, 4, 4, 13), (4992, 1, 96, 24, 384), 0); del buf1130  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_45.run(buf1132, buf1133, 4992, 121, grid=grid(4992), stream=stream0)
        buf1134 = empty((1, 24, 4, 4), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_46.run(buf1133, buf1134, 384, 13, grid=grid(384), stream=stream0)
        del buf1133
        buf1135 = reinterpret_tensor(buf1122, (8, 24, 56, 56), (75264, 3136, 56, 1), 0); del buf1122  # reuse
        # Source Nodes: [], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_47.run(buf1135, 602112, grid=grid(602112), stream=stream0)
        buf1136 = reinterpret_tensor(buf1121, (1568, 24, 4, 4), (384, 16, 4, 1), 0); del buf1121  # reuse
        buf1137 = reinterpret_tensor(buf1136, (8, 24, 4, 14, 4, 14), (75264, 16, 4, 5376, 1, 384), 0); del buf1136  # reuse
        # Source Nodes: [], Original ATen: [aten.clone, aten.col2im]
        triton_poi_fused_clone_col2im_48.run(buf1137, buf1132, 37632, 16, grid=grid(37632, 16), stream=stream0)
        del buf1132
        aten.index_put_(buf1135, [None, None, unsqueeze_5, add], buf1137, True)
        del add
        del buf1137
        del unsqueeze_5
        buf1140 = empty_strided((24, 4), (1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_49.run(buf1135, buf1140, 96, 6272, grid=grid(96), stream=stream0)
        buf1141 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_50.run(buf1140, buf1141, 24, 4, grid=grid(24), stream=stream0)
        del buf1140
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1142 = aten.convolution_backward(buf1135, primals_352, primals_4, [24], [4, 4], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf1135
        del primals_352
        del primals_4
        buf1143 = buf1142[1]
        return (buf1134, buf1114, buf1113, buf1143, buf1141, buf1129, buf1131, reinterpret_tensor(buf1123, (384, 384), (384, 1), 0), reinterpret_tensor(buf1125, (384, ), (1, ), 0), buf1118, buf1120, buf1109, buf1111, reinterpret_tensor(buf1104, (48, 24), (24, 1), 0), reinterpret_tensor(buf1101, (24, 24), (24, 1), 0), reinterpret_tensor(buf1090, (24, 24), (24, 1), 0), reinterpret_tensor(buf1092, (24, ), (1, ), 0), buf1085, buf1087, reinterpret_tensor(buf1079, (96, 24), (24, 1), 0), reinterpret_tensor(buf1081, (96, ), (1, ), 0), reinterpret_tensor(buf1074, (24, 96), (96, 1), 0), reinterpret_tensor(buf1076, (24, ), (1, ), 0), buf1069, buf1071, reinterpret_tensor(buf1063, (384, 384), (384, 1), 0), reinterpret_tensor(buf1065, (384, ), (1, ), 0), buf1057, buf1059, reinterpret_tensor(buf1052, (768, 384), (384, 1), 0), reinterpret_tensor(buf1049, (384, 384), (384, 1), 0), reinterpret_tensor(buf1038, (384, 384), (384, 1), 0), reinterpret_tensor(buf1040, (384, ), (1, ), 0), buf1033, buf1035, reinterpret_tensor(buf1027, (1536, 384), (384, 1), 0), reinterpret_tensor(buf1029, (1536, ), (1, ), 0), reinterpret_tensor(buf1022, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf1024, (384, ), (1, ), 0), buf1017, buf1019, reinterpret_tensor(buf1012, (48, 24), (24, 1), 0), reinterpret_tensor(buf1009, (24, 24), (24, 1), 0), reinterpret_tensor(buf998, (24, 24), (24, 1), 0), reinterpret_tensor(buf1000, (24, ), (1, ), 0), buf993, buf995, reinterpret_tensor(buf987, (96, 24), (24, 1), 0), reinterpret_tensor(buf989, (96, ), (1, ), 0), reinterpret_tensor(buf982, (24, 96), (96, 1), 0), reinterpret_tensor(buf984, (24, ), (1, ), 0), buf977, buf979, reinterpret_tensor(buf971, (384, 384), (384, 1), 0), reinterpret_tensor(buf973, (384, ), (1, ), 0), buf965, buf967, reinterpret_tensor(buf960, (768, 384), (384, 1), 0), reinterpret_tensor(buf957, (384, 384), (384, 1), 0), reinterpret_tensor(buf946, (384, 384), (384, 1), 0), reinterpret_tensor(buf948, (384, ), (1, ), 0), buf941, buf943, reinterpret_tensor(buf935, (1536, 384), (384, 1), 0), reinterpret_tensor(buf937, (1536, ), (1, ), 0), reinterpret_tensor(buf930, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf932, (384, ), (1, ), 0), buf925, buf927, reinterpret_tensor(buf920, (48, 24), (24, 1), 0), reinterpret_tensor(buf917, (24, 24), (24, 1), 0), reinterpret_tensor(buf906, (24, 24), (24, 1), 0), reinterpret_tensor(buf908, (24, ), (1, ), 0), buf901, buf903, reinterpret_tensor(buf895, (96, 24), (24, 1), 0), reinterpret_tensor(buf897, (96, ), (1, ), 0), reinterpret_tensor(buf890, (24, 96), (96, 1), 0), reinterpret_tensor(buf892, (24, ), (1, ), 0), buf885, buf887, reinterpret_tensor(buf879, (384, 384), (384, 1), 0), reinterpret_tensor(buf881, (384, ), (1, ), 0), buf873, buf875, reinterpret_tensor(buf868, (768, 384), (384, 1), 0), reinterpret_tensor(buf865, (384, 384), (384, 1), 0), reinterpret_tensor(buf854, (384, 384), (384, 1), 0), reinterpret_tensor(buf856, (384, ), (1, ), 0), buf849, buf851, reinterpret_tensor(buf843, (1536, 384), (384, 1), 0), reinterpret_tensor(buf845, (1536, ), (1, ), 0), reinterpret_tensor(buf838, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf840, (384, ), (1, ), 0), buf833, buf835, reinterpret_tensor(buf828, (48, 24), (24, 1), 0), reinterpret_tensor(buf825, (24, 24), (24, 1), 0), reinterpret_tensor(buf814, (24, 24), (24, 1), 0), reinterpret_tensor(buf816, (24, ), (1, ), 0), buf809, buf811, reinterpret_tensor(buf803, (96, 24), (24, 1), 0), reinterpret_tensor(buf805, (96, ), (1, ), 0), reinterpret_tensor(buf798, (24, 96), (96, 1), 0), reinterpret_tensor(buf800, (24, ), (1, ), 0), buf793, buf795, reinterpret_tensor(buf787, (384, 384), (384, 1), 0), reinterpret_tensor(buf789, (384, ), (1, ), 0), buf781, buf783, reinterpret_tensor(buf776, (768, 384), (384, 1), 0), reinterpret_tensor(buf773, (384, 384), (384, 1), 0), reinterpret_tensor(buf762, (384, 384), (384, 1), 0), reinterpret_tensor(buf764, (384, ), (1, ), 0), buf757, buf759, reinterpret_tensor(buf751, (1536, 384), (384, 1), 0), reinterpret_tensor(buf753, (1536, ), (1, ), 0), reinterpret_tensor(buf746, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf748, (384, ), (1, ), 0), buf741, buf743, reinterpret_tensor(buf736, (48, 24), (24, 1), 0), reinterpret_tensor(buf733, (24, 24), (24, 1), 0), reinterpret_tensor(buf722, (24, 24), (24, 1), 0), reinterpret_tensor(buf724, (24, ), (1, ), 0), buf717, buf719, reinterpret_tensor(buf711, (96, 24), (24, 1), 0), reinterpret_tensor(buf713, (96, ), (1, ), 0), reinterpret_tensor(buf706, (24, 96), (96, 1), 0), reinterpret_tensor(buf708, (24, ), (1, ), 0), buf701, buf703, reinterpret_tensor(buf695, (384, 384), (384, 1), 0), reinterpret_tensor(buf697, (384, ), (1, ), 0), buf689, buf691, reinterpret_tensor(buf684, (768, 384), (384, 1), 0), reinterpret_tensor(buf681, (384, 384), (384, 1), 0), reinterpret_tensor(buf670, (384, 384), (384, 1), 0), reinterpret_tensor(buf672, (384, ), (1, ), 0), buf665, buf667, reinterpret_tensor(buf659, (1536, 384), (384, 1), 0), reinterpret_tensor(buf661, (1536, ), (1, ), 0), reinterpret_tensor(buf654, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf656, (384, ), (1, ), 0), buf649, buf651, reinterpret_tensor(buf644, (48, 24), (24, 1), 0), reinterpret_tensor(buf641, (24, 24), (24, 1), 0), reinterpret_tensor(buf630, (24, 24), (24, 1), 0), reinterpret_tensor(buf632, (24, ), (1, ), 0), buf625, buf627, reinterpret_tensor(buf619, (96, 24), (24, 1), 0), reinterpret_tensor(buf621, (96, ), (1, ), 0), reinterpret_tensor(buf614, (24, 96), (96, 1), 0), reinterpret_tensor(buf616, (24, ), (1, ), 0), buf609, buf611, reinterpret_tensor(buf603, (384, 384), (384, 1), 0), reinterpret_tensor(buf605, (384, ), (1, ), 0), buf597, buf599, reinterpret_tensor(buf592, (768, 384), (384, 1), 0), reinterpret_tensor(buf589, (384, 384), (384, 1), 0), reinterpret_tensor(buf578, (384, 384), (384, 1), 0), reinterpret_tensor(buf580, (384, ), (1, ), 0), buf573, buf575, reinterpret_tensor(buf567, (1536, 384), (384, 1), 0), reinterpret_tensor(buf569, (1536, ), (1, ), 0), reinterpret_tensor(buf562, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf564, (384, ), (1, ), 0), buf557, buf559, reinterpret_tensor(buf552, (48, 24), (24, 1), 0), reinterpret_tensor(buf549, (24, 24), (24, 1), 0), reinterpret_tensor(buf538, (24, 24), (24, 1), 0), reinterpret_tensor(buf540, (24, ), (1, ), 0), buf533, buf535, reinterpret_tensor(buf527, (96, 24), (24, 1), 0), reinterpret_tensor(buf529, (96, ), (1, ), 0), reinterpret_tensor(buf522, (24, 96), (96, 1), 0), reinterpret_tensor(buf524, (24, ), (1, ), 0), buf517, buf519, reinterpret_tensor(buf511, (384, 384), (384, 1), 0), reinterpret_tensor(buf513, (384, ), (1, ), 0), buf505, buf507, reinterpret_tensor(buf500, (768, 384), (384, 1), 0), reinterpret_tensor(buf497, (384, 384), (384, 1), 0), reinterpret_tensor(buf486, (384, 384), (384, 1), 0), reinterpret_tensor(buf488, (384, ), (1, ), 0), buf481, buf483, reinterpret_tensor(buf475, (1536, 384), (384, 1), 0), reinterpret_tensor(buf477, (1536, ), (1, ), 0), reinterpret_tensor(buf470, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf472, (384, ), (1, ), 0), buf465, buf467, reinterpret_tensor(buf460, (48, 24), (24, 1), 0), reinterpret_tensor(buf457, (24, 24), (24, 1), 0), reinterpret_tensor(buf446, (24, 24), (24, 1), 0), reinterpret_tensor(buf448, (24, ), (1, ), 0), buf441, buf443, reinterpret_tensor(buf435, (96, 24), (24, 1), 0), reinterpret_tensor(buf437, (96, ), (1, ), 0), reinterpret_tensor(buf430, (24, 96), (96, 1), 0), reinterpret_tensor(buf432, (24, ), (1, ), 0), buf425, buf427, reinterpret_tensor(buf419, (384, 384), (384, 1), 0), reinterpret_tensor(buf421, (384, ), (1, ), 0), buf413, buf415, reinterpret_tensor(buf408, (768, 384), (384, 1), 0), reinterpret_tensor(buf405, (384, 384), (384, 1), 0), reinterpret_tensor(buf394, (384, 384), (384, 1), 0), reinterpret_tensor(buf396, (384, ), (1, ), 0), buf389, buf391, reinterpret_tensor(buf383, (1536, 384), (384, 1), 0), reinterpret_tensor(buf385, (1536, ), (1, ), 0), reinterpret_tensor(buf378, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf380, (384, ), (1, ), 0), buf373, buf375, reinterpret_tensor(buf368, (48, 24), (24, 1), 0), reinterpret_tensor(buf365, (24, 24), (24, 1), 0), reinterpret_tensor(buf354, (24, 24), (24, 1), 0), reinterpret_tensor(buf356, (24, ), (1, ), 0), buf349, buf351, reinterpret_tensor(buf343, (96, 24), (24, 1), 0), reinterpret_tensor(buf345, (96, ), (1, ), 0), reinterpret_tensor(buf338, (24, 96), (96, 1), 0), reinterpret_tensor(buf340, (24, ), (1, ), 0), buf333, buf335, reinterpret_tensor(buf327, (384, 384), (384, 1), 0), reinterpret_tensor(buf329, (384, ), (1, ), 0), buf321, buf323, reinterpret_tensor(buf316, (768, 384), (384, 1), 0), reinterpret_tensor(buf313, (384, 384), (384, 1), 0), reinterpret_tensor(buf302, (384, 384), (384, 1), 0), reinterpret_tensor(buf304, (384, ), (1, ), 0), buf297, buf299, reinterpret_tensor(buf291, (1536, 384), (384, 1), 0), reinterpret_tensor(buf293, (1536, ), (1, ), 0), reinterpret_tensor(buf286, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf288, (384, ), (1, ), 0), buf281, buf283, reinterpret_tensor(buf276, (48, 24), (24, 1), 0), reinterpret_tensor(buf273, (24, 24), (24, 1), 0), reinterpret_tensor(buf262, (24, 24), (24, 1), 0), reinterpret_tensor(buf264, (24, ), (1, ), 0), buf257, buf259, reinterpret_tensor(buf251, (96, 24), (24, 1), 0), reinterpret_tensor(buf253, (96, ), (1, ), 0), reinterpret_tensor(buf246, (24, 96), (96, 1), 0), reinterpret_tensor(buf248, (24, ), (1, ), 0), buf241, buf243, reinterpret_tensor(buf235, (384, 384), (384, 1), 0), reinterpret_tensor(buf237, (384, ), (1, ), 0), buf229, buf231, reinterpret_tensor(buf224, (768, 384), (384, 1), 0), reinterpret_tensor(buf221, (384, 384), (384, 1), 0), reinterpret_tensor(buf210, (384, 384), (384, 1), 0), reinterpret_tensor(buf212, (384, ), (1, ), 0), buf205, buf207, reinterpret_tensor(buf199, (1536, 384), (384, 1), 0), reinterpret_tensor(buf201, (1536, ), (1, ), 0), reinterpret_tensor(buf194, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf196, (384, ), (1, ), 0), buf189, buf191, reinterpret_tensor(buf184, (48, 24), (24, 1), 0), reinterpret_tensor(buf181, (24, 24), (24, 1), 0), reinterpret_tensor(buf170, (24, 24), (24, 1), 0), reinterpret_tensor(buf172, (24, ), (1, ), 0), buf165, buf167, reinterpret_tensor(buf159, (96, 24), (24, 1), 0), reinterpret_tensor(buf161, (96, ), (1, ), 0), reinterpret_tensor(buf154, (24, 96), (96, 1), 0), reinterpret_tensor(buf156, (24, ), (1, ), 0), buf149, buf151, reinterpret_tensor(buf143, (384, 384), (384, 1), 0), reinterpret_tensor(buf145, (384, ), (1, ), 0), buf137, buf139, reinterpret_tensor(buf132, (768, 384), (384, 1), 0), reinterpret_tensor(buf129, (384, 384), (384, 1), 0), reinterpret_tensor(buf118, (384, 384), (384, 1), 0), reinterpret_tensor(buf120, (384, ), (1, ), 0), buf113, buf115, reinterpret_tensor(buf107, (1536, 384), (384, 1), 0), reinterpret_tensor(buf109, (1536, ), (1, ), 0), reinterpret_tensor(buf102, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf104, (384, ), (1, ), 0), buf97, buf99, reinterpret_tensor(buf92, (48, 24), (24, 1), 0), reinterpret_tensor(buf89, (24, 24), (24, 1), 0), reinterpret_tensor(buf78, (24, 24), (24, 1), 0), reinterpret_tensor(buf80, (24, ), (1, ), 0), buf73, buf75, reinterpret_tensor(buf67, (96, 24), (24, 1), 0), reinterpret_tensor(buf69, (96, ), (1, ), 0), reinterpret_tensor(buf62, (24, 96), (96, 1), 0), reinterpret_tensor(buf64, (24, ), (1, ), 0), buf58, buf60, reinterpret_tensor(buf51, (384, 384), (384, 1), 0), reinterpret_tensor(buf53, (384, ), (1, ), 0), buf45, buf47, reinterpret_tensor(buf40, (768, 384), (384, 1), 0), reinterpret_tensor(buf37, (384, 384), (384, 1), 0), reinterpret_tensor(buf26, (384, 384), (384, 1), 0), reinterpret_tensor(buf28, (384, ), (1, ), 0), buf21, buf23, reinterpret_tensor(buf15, (1536, 384), (384, 1), 0), reinterpret_tensor(buf17, (1536, ), (1, ), 0), reinterpret_tensor(buf10, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf12, (384, ), (1, ), 0), buf7, buf8, reinterpret_tensor(buf1, (1000, 384), (384, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((24, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    add = rand_strided((4, 14), (14, 1), device='cuda:0', dtype=torch.int64)
    unsqueeze_5 = rand_strided((4, 14, 1, 1), (14, 1, 1, 1), device='cuda:0', dtype=torch.int64)
    clone_2 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    view_4 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    getitem_5 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_2 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    view_6 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    mul_7 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    mul_12 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_26 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_5 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_28 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_41 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_17 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    view_43 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_6 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_45 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_47 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    view_60 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    mul_25 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_62 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    addmm_9 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    view_64 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    mul_30 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_67 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_27 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_10 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_69 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_82 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_35 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    view_84 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_13 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    view_101 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    mul_43 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_103 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    view_105 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    mul_48 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_108 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    cat_3 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_41 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_15 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_123 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_53 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    view_125 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_20 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_127 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_129 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    view_142 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    mul_61 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_144 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    addmm_23 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    view_146 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    mul_66 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_149 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    cat_4 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_55 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_20 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_151 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_164 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_71 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    view_166 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_27 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_168 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_170 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    view_183 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    mul_79 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_185 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    addmm_30 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    view_187 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    mul_84 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_190 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    cat_5 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_69 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_25 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_192 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_205 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_89 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    view_207 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_209 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_211 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    view_224 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    mul_97 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_226 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    addmm_37 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    view_228 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    mul_102 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_231 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    cat_6 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_83 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_30 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_233 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_246 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_107 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    view_248 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_41 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_250 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_252 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    view_265 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    mul_115 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_267 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    addmm_44 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    view_269 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    mul_120 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_272 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    cat_7 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_97 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_35 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_274 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_287 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_125 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    view_289 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_48 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_291 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_293 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    view_306 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    mul_133 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_308 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    addmm_51 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    view_310 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    mul_138 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_313 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    cat_8 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_111 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_40 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_315 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_328 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_143 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    view_330 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_55 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_332 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_334 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    view_347 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    mul_151 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_349 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    addmm_58 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    view_351 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    mul_156 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_354 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    cat_9 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_125 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_45 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_356 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_369 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_161 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    view_371 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_62 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_373 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_375 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    view_388 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    mul_169 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_390 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    addmm_65 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    view_392 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    mul_174 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_395 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    cat_10 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_139 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_50 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_397 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_410 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_179 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    view_412 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_69 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_414 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_416 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    view_429 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    mul_187 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_431 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    addmm_72 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    view_433 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    mul_192 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_436 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    cat_11 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_153 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_55 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_438 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_451 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_197 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    view_453 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_76 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_455 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_457 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    view_470 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    mul_205 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_472 = rand_strided((25088, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    addmm_79 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    view_474 = rand_strided((25088, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    mul_210 = rand_strided((1568, 16, 24), (384, 24, 1), device='cuda:0', dtype=torch.float32)
    view_477 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    cat_12 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_167 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_60 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_479 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_492 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_215 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    view_494 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_83 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_496 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    mul_220 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    clone_184 = rand_strided((8, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_233 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_237 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_241 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_245 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_250 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_251 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_24 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    permute_252 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_253 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_258 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_263 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_265 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_269 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    permute_273 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_277 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_282 = rand_strided((6272, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_283 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    alias_25 = rand_strided((1568, 4, 16, 16), (1024, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_284 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    permute_285 = rand_strided((6272, 16, 6), (96, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_290 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_295 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_29 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_297 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_301 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_305 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_310 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_311 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_26 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    permute_312 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_313 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_318 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_323 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_325 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_329 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    permute_333 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_337 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_342 = rand_strided((6272, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_343 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    alias_27 = rand_strided((1568, 4, 16, 16), (1024, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_344 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    permute_345 = rand_strided((6272, 16, 6), (96, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_350 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_355 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_357 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_361 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_35 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_365 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_370 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_371 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_28 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    permute_372 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_373 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_378 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_383 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_385 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_389 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    permute_393 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_38 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_397 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_402 = rand_strided((6272, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_403 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    alias_29 = rand_strided((1568, 4, 16, 16), (1024, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_404 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    permute_405 = rand_strided((6272, 16, 6), (96, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_410 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_415 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_417 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_421 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_425 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_430 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_431 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_30 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    permute_432 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_433 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_438 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_443 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_445 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_449 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    permute_453 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_457 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_462 = rand_strided((6272, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_463 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    alias_31 = rand_strided((1568, 4, 16, 16), (1024, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_464 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    permute_465 = rand_strided((6272, 16, 6), (96, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_470 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_475 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_44 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_477 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_481 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_485 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_490 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_491 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_32 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    permute_492 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_493 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_498 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_503 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_505 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_509 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    permute_513 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_517 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_522 = rand_strided((6272, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_523 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    alias_33 = rand_strided((1568, 4, 16, 16), (1024, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_524 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    permute_525 = rand_strided((6272, 16, 6), (96, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_530 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_535 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_537 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_541 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_50 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_545 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_550 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_551 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_34 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    permute_552 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_553 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_558 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_563 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_565 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_569 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    permute_573 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_53 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_577 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_582 = rand_strided((6272, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_583 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    alias_35 = rand_strided((1568, 4, 16, 16), (1024, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_584 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    permute_585 = rand_strided((6272, 16, 6), (96, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_590 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_595 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_597 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_601 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_605 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_610 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_611 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_36 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    permute_612 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_613 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_618 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_623 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_625 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_629 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    permute_633 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_637 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_642 = rand_strided((6272, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_643 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    alias_37 = rand_strided((1568, 4, 16, 16), (1024, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_644 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    permute_645 = rand_strided((6272, 16, 6), (96, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_650 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_655 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_59 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_657 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_661 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_665 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_670 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_671 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_38 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    permute_672 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_673 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_678 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_683 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_685 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_689 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    permute_693 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_63 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_697 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_702 = rand_strided((6272, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_703 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    alias_39 = rand_strided((1568, 4, 16, 16), (1024, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_704 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    permute_705 = rand_strided((6272, 16, 6), (96, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_710 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_715 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_64 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_717 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_721 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_65 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_725 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_730 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_731 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_40 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    permute_732 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_733 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_738 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_743 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_745 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_749 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    permute_753 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_68 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_757 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_762 = rand_strided((6272, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_763 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    alias_41 = rand_strided((1568, 4, 16, 16), (1024, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_764 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    permute_765 = rand_strided((6272, 16, 6), (96, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_770 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_775 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_69 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_777 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_781 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_70 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_785 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_790 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_791 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_42 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    permute_792 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_793 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_798 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_803 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_805 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_809 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    permute_813 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_73 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_817 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_822 = rand_strided((6272, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_823 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    alias_43 = rand_strided((1568, 4, 16, 16), (1024, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_824 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    permute_825 = rand_strided((6272, 16, 6), (96, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_830 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_835 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_74 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_837 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_841 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_75 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_845 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_850 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_851 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_44 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    permute_852 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_853 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_858 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_863 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_865 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_869 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    permute_873 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_78 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_877 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_882 = rand_strided((6272, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_883 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    alias_45 = rand_strided((1568, 4, 16, 16), (1024, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_884 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    permute_885 = rand_strided((6272, 16, 6), (96, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_890 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_895 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_79 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_897 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_901 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_80 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_905 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_910 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_911 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_46 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    permute_912 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_913 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_918 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_923 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_925 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_929 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    permute_933 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    div_83 = rand_strided((1568, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_937 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_942 = rand_strided((6272, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_943 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    alias_47 = rand_strided((1568, 4, 16, 16), (1024, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_944 = rand_strided((6272, 6, 16), (96, 1, 6), device='cuda:0', dtype=torch.float32)
    permute_945 = rand_strided((6272, 16, 6), (96, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_950 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_955 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    permute_957 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_6, primals_10, primals_12, primals_18, primals_24, primals_28, primals_34, primals_40, primals_46, primals_52, primals_56, primals_62, primals_68, primals_74, primals_80, primals_84, primals_90, primals_96, primals_102, primals_108, primals_112, primals_118, primals_124, primals_130, primals_136, primals_140, primals_146, primals_152, primals_158, primals_164, primals_168, primals_174, primals_180, primals_186, primals_192, primals_196, primals_202, primals_208, primals_214, primals_220, primals_224, primals_230, primals_236, primals_242, primals_248, primals_252, primals_258, primals_264, primals_270, primals_276, primals_280, primals_286, primals_292, primals_298, primals_304, primals_308, primals_314, primals_320, primals_326, primals_332, primals_336, primals_342, primals_348, primals_352, add, unsqueeze_5, clone_2, getitem_1, rsqrt, view_4, addmm, getitem_3, rsqrt_1, getitem_5, rsqrt_2, view_6, view_19, mul_7, view_21, addmm_2, view_23, mul_12, view_26, cat_1, getitem_13, rsqrt_5, view_28, view_41, mul_17, view_43, addmm_6, view_45, view_47, view_60, mul_25, view_62, addmm_9, view_64, mul_30, view_67, cat_2, getitem_27, rsqrt_10, view_69, view_82, mul_35, view_84, addmm_13, view_86, view_88, view_101, mul_43, view_103, addmm_16, view_105, mul_48, view_108, cat_3, getitem_41, rsqrt_15, view_110, view_123, mul_53, view_125, addmm_20, view_127, view_129, view_142, mul_61, view_144, addmm_23, view_146, mul_66, view_149, cat_4, getitem_55, rsqrt_20, view_151, view_164, mul_71, view_166, addmm_27, view_168, view_170, view_183, mul_79, view_185, addmm_30, view_187, mul_84, view_190, cat_5, getitem_69, rsqrt_25, view_192, view_205, mul_89, view_207, addmm_34, view_209, view_211, view_224, mul_97, view_226, addmm_37, view_228, mul_102, view_231, cat_6, getitem_83, rsqrt_30, view_233, view_246, mul_107, view_248, addmm_41, view_250, view_252, view_265, mul_115, view_267, addmm_44, view_269, mul_120, view_272, cat_7, getitem_97, rsqrt_35, view_274, view_287, mul_125, view_289, addmm_48, view_291, view_293, view_306, mul_133, view_308, addmm_51, view_310, mul_138, view_313, cat_8, getitem_111, rsqrt_40, view_315, view_328, mul_143, view_330, addmm_55, view_332, view_334, view_347, mul_151, view_349, addmm_58, view_351, mul_156, view_354, cat_9, getitem_125, rsqrt_45, view_356, view_369, mul_161, view_371, addmm_62, view_373, view_375, view_388, mul_169, view_390, addmm_65, view_392, mul_174, view_395, cat_10, getitem_139, rsqrt_50, view_397, view_410, mul_179, view_412, addmm_69, view_414, view_416, view_429, mul_187, view_431, addmm_72, view_433, mul_192, view_436, cat_11, getitem_153, rsqrt_55, view_438, view_451, mul_197, view_453, addmm_76, view_455, view_457, view_470, mul_205, view_472, addmm_79, view_474, mul_210, view_477, cat_12, getitem_167, rsqrt_60, view_479, view_492, mul_215, view_494, addmm_83, view_496, mul_220, clone_184, permute_233, div_24, permute_237, permute_241, div_25, permute_245, permute_250, permute_251, alias_24, permute_252, permute_253, permute_258, permute_263, permute_265, div_27, permute_269, permute_273, div_28, permute_277, permute_282, permute_283, alias_25, permute_284, permute_285, permute_290, permute_295, div_29, permute_297, permute_301, div_30, permute_305, permute_310, permute_311, alias_26, permute_312, permute_313, permute_318, permute_323, permute_325, permute_329, permute_333, div_33, permute_337, permute_342, permute_343, alias_27, permute_344, permute_345, permute_350, permute_355, div_34, permute_357, permute_361, div_35, permute_365, permute_370, permute_371, alias_28, permute_372, permute_373, permute_378, permute_383, permute_385, permute_389, permute_393, div_38, permute_397, permute_402, permute_403, alias_29, permute_404, permute_405, permute_410, permute_415, div_39, permute_417, permute_421, div_40, permute_425, permute_430, permute_431, alias_30, permute_432, permute_433, permute_438, permute_443, permute_445, permute_449, permute_453, div_43, permute_457, permute_462, permute_463, alias_31, permute_464, permute_465, permute_470, permute_475, div_44, permute_477, permute_481, div_45, permute_485, permute_490, permute_491, alias_32, permute_492, permute_493, permute_498, permute_503, permute_505, permute_509, permute_513, div_48, permute_517, permute_522, permute_523, alias_33, permute_524, permute_525, permute_530, permute_535, div_49, permute_537, permute_541, div_50, permute_545, permute_550, permute_551, alias_34, permute_552, permute_553, permute_558, permute_563, permute_565, permute_569, permute_573, div_53, permute_577, permute_582, permute_583, alias_35, permute_584, permute_585, permute_590, permute_595, div_54, permute_597, permute_601, div_55, permute_605, permute_610, permute_611, alias_36, permute_612, permute_613, permute_618, permute_623, permute_625, permute_629, permute_633, div_58, permute_637, permute_642, permute_643, alias_37, permute_644, permute_645, permute_650, permute_655, div_59, permute_657, permute_661, div_60, permute_665, permute_670, permute_671, alias_38, permute_672, permute_673, permute_678, permute_683, permute_685, permute_689, permute_693, div_63, permute_697, permute_702, permute_703, alias_39, permute_704, permute_705, permute_710, permute_715, div_64, permute_717, permute_721, div_65, permute_725, permute_730, permute_731, alias_40, permute_732, permute_733, permute_738, permute_743, permute_745, permute_749, permute_753, div_68, permute_757, permute_762, permute_763, alias_41, permute_764, permute_765, permute_770, permute_775, div_69, permute_777, permute_781, div_70, permute_785, permute_790, permute_791, alias_42, permute_792, permute_793, permute_798, permute_803, permute_805, permute_809, permute_813, div_73, permute_817, permute_822, permute_823, alias_43, permute_824, permute_825, permute_830, permute_835, div_74, permute_837, permute_841, div_75, permute_845, permute_850, permute_851, alias_44, permute_852, permute_853, permute_858, permute_863, permute_865, permute_869, permute_873, div_78, permute_877, permute_882, permute_883, alias_45, permute_884, permute_885, permute_890, permute_895, div_79, permute_897, permute_901, div_80, permute_905, permute_910, permute_911, alias_46, permute_912, permute_913, permute_918, permute_923, permute_925, permute_929, permute_933, div_83, permute_937, permute_942, permute_943, alias_47, permute_944, permute_945, permute_950, permute_955, permute_957, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tnt_s_patch16_224', benchmark_compiled_module)
