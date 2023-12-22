
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


# kernel path: /tmp/torchinductor_youkaichao/jh/cjhb7qtpf546pjhsfrzu45dwenhmgfms2fuwq25y2vv4wzr6f7x7.py
# Source Nodes: [], Original ATen: [aten.div, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_div_leaky_relu_backward_native_batch_norm_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_leaky_relu_backward_native_batch_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*(r2 // 64)) + (2048*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr2 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 64.0
        tmp3 = tmp1 / tmp2
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp0, tmp3, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4l/c4lygknlqblqorvb3bn64yutagrs4s3aweujksxihmlkl62l4v67.py
# Source Nodes: [], Original ATen: [aten.div, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_div_leaky_relu_backward_native_batch_norm_backward_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_leaky_relu_backward_native_batch_norm_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/24/c24agrpnw2pdxe7ezeod5dc5aj4w3hbdvj23b55k2axtq7bvq5rt.py
# Source Nodes: [], Original ATen: [aten.div, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_div_leaky_relu_backward_native_batch_norm_backward_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_leaky_relu_backward_native_batch_norm_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rs/crsd274udqcphbmssysgctnxbq5sx5ufqeyhu4gumccwzw6zh73r.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_div_leaky_relu_backward_native_batch_norm_backward_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_leaky_relu_backward_native_batch_norm_backward_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1024
    x2 = (xindex // 65536)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x3), None)
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = 64.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.01
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp0, tmp3, tmp5)
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
    tl.store(out_ptr0 + (x3), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cg/ccgtfhpdbijq7sxevmgrdyxtn3p4uzuukkrl4dqbamvkjlw6ii6d.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (32768 + (64*x0) + (65536*(r2 // 64)) + (131072*x1) + (r2 % 64)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pr/cpr2upn2mqcaedcptruehvkdfbc23hu5zeuhekyo5gmlject3j2j.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_leaky_relu_backward_native_batch_norm_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_leaky_relu_backward_native_batch_norm_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uu/cuubufxdvtpq5nlpkkf5wtyperq6sbtmtccu7jb6wfkxrcyn4ejc.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_leaky_relu_backward_native_batch_norm_backward_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_leaky_relu_backward_native_batch_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vo/cvo6knciczqp5qqgk2wij7klsgnf2k4nkqqis6bw23cxiw6zctqi.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (x2 + (512*y3)), xmask & ymask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (32768 + y0 + (64*x2) + (65536*y1)), xmask & ymask)
    tmp5 = tl.load(in_ptr2 + (x2 + (512*y3)), xmask & ymask)
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tl.where(tmp0, tmp1, tmp3)
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
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i2/ci235ax4ulzotmqarxm4q3p2cpggdjspyaqevnfqnpixys6whdsf.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((64*x0) + (32768*(r2 // 64)) + (65536*x1) + (r2 % 64)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3o/c3oo7iqzcxo4yer6bzwozeh3btoakq2xjjx5uijbww3y66tydxc5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (x2 + (512*y3)), xmask & ymask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (64*x2) + (32768*y1)), xmask & ymask)
    tmp5 = tl.load(in_ptr2 + (x2 + (512*y3)), xmask & ymask)
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tl.where(tmp0, tmp1, tmp3)
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
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xu/cxurxbtcwzvbzrojbfhwtwm5vqyrk7yozapbrv6wiyisxf4tzpf7.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((64*x0) + (32768*(r2 // 64)) + (65536*x1) + (r2 % 64)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yn/cynpaposlzb4g7zoqr7adx7tq5tysyvcmgbktwjts5xczk73bhao.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (x2 + (512*y3)), xmask & ymask)
    tmp3 = tl.load(in_ptr1 + (y0 + (64*x2) + (32768*y1)), xmask & ymask)
    tmp7 = tl.load(in_ptr2 + (x2 + (512*y3)), xmask & ymask)
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.01
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp2, tmp3, tmp5)
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
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wl/cwlmoatuphzl43egsaabfgye3qlli2y57mqf57ypnf2nfppgrdqc.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel):
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r3)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = 0.01
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp0, tmp3, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/px/cpxqpv3c3pprhpi2pwiinlh2ytdk7sfpy4qhvvsk6mxfpuuxcucf.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((64*x0) + (32768*(r2 // 64)) + (65536*x1) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + ((64*x0) + (32768*(r2 // 64)) + (65536*x1) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp0, tmp3, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zw/czwc6ivvoloizcy4b7rfseotaz4e5xx4bff7zu4zkeu7hilxotfi.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_leaky_relu_backward_native_batch_norm_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_backward_native_batch_norm_backward_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (x2 + (512*y3)), xmask & ymask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (64*x2) + (32768*y1)), xmask & ymask)
    tmp2 = tl.load(in_ptr2 + (y0 + (64*x2) + (32768*y1)), xmask & ymask)
    tmp7 = tl.load(in_ptr3 + (x2 + (512*y3)), xmask & ymask)
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.01
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp0, tmp3, tmp5)
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
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sb/csbqr7jh3lkgexd35zlbuzhqhkq4yzv2xyjmfli6j64u54rwl2bz.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r3)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr3 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr4 + (x0 + (512*r3)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.01
    tmp7 = tmp5 * tmp6
    tmp8 = tl.where(tmp0, tmp5, tmp7)
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = tmp13 - tmp14
    tmp16 = tmp8 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = tmp20 * tmp21
    tl.store(out_ptr2 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tl.store(out_ptr1 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tg/ctgmflvpd3bznovkq7wdx4q3asd4d74iezljgpgph3cr6mnsdw53.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (32768*y1)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0 + (512*x2) + (32768*y1)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.01
    tmp7 = tmp5 * tmp6
    tmp8 = tl.where(tmp0, tmp5, tmp7)
    tmp11 = tmp9 - tmp10
    tmp13 = 0.001953125
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tl.store(out_ptr1 + (y0 + (512*x2) + (32768*y1)), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/np/cnpee66obm6jnslk7zc6yk2wfbpgrey2cgqumtrwov6asz5sjwzv.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward]

triton_poi_fused_add_leaky_relu_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_backward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (32768*y1)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.01
    tmp9 = tmp7 * tmp8
    tmp10 = tl.where(tmp0, tmp7, tmp9)
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4o/c4oh7qikvymesr6carzlce3uehlsp4vk3k45kiyjim3byd2yqdeb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3v/c3valqx3kzwcergvf7nrsw5qm2ssffholg5gvgolnuuranlwu7hx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*x0) + (32768*(r2 // 64)) + (65536*x1) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/63/c636p6n4gurkocchnioni3k6nyx4qlaeuvhba5c4epa2i633q2bj.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (32768*y1)), xmask & ymask)
    tmp1 = tl.load(in_ptr1 + (x2 + (512*y3)), xmask & ymask)
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
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zv/czvcqq3szdfkpeyx6cgy4guujgv5yean4axyawlg26f3oteo3jgq.py
# Source Nodes: [], Original ATen: [aten.cat]

triton_poi_fused_cat_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64) % 1024
    x3 = xindex
    x2 = (xindex // 65536)
    x4 = xindex % 65536
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_out_ptr0 + (x3), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 1024, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr0 + ((-32768) + x4 + (32768*x2)), tmp8, other=0.0)
    tmp12 = tl.load(in_ptr1 + ((-32768) + x4 + (32768*x2)), tmp8, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr2 + ((-32768) + x4 + (32768*x2)), tmp8, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.load(in_ptr3 + ((-32768) + x4 + (32768*x2)), tmp8, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr4 + ((-32768) + x4 + (32768*x2)), tmp8, other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp8, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(in_out_ptr0 + (x3), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3e/c3exhvuf7nkztzpaex2bpkcmstztdpbob2v3vrt7k6bxyzknm7na.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((64*x0) + (65536*(r2 // 64)) + (131072*x1) + (r2 % 64)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nn/cnnzribusc65ynkv6xbgxkorxjcforabk35w54fj2c2afqqsbj47.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (64*x2) + (65536*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tl.where(tmp0, tmp1, tmp3)
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
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hx/chxilga52hkevgrukk4gdpe6nnndq33kjwgm7yfph2czqtknvb76.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((64*x0) + (65536*(r2 // 64)) + (131072*x1) + (r2 % 64)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr2 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7d/c7d2w47jvumly3nt7s7mkg7snn6m4e3ffs3zem6exw534agg3qbh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (64*x2) + (65536*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.01
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp2, tmp3, tmp5)
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
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/la/clauf74nonwhantcvackzivzbwhylcey6bmlf7mm2ny2lqxpsnzd.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x1 + (512*r2) + (65536*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (131072*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fj/cfj6rzyef2rnuc6hhx6thlu4ffllqrbbql76tm3cf4am5cxldzxn.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_leaky_relu_backward_native_batch_norm_backward_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_leaky_relu_backward_native_batch_norm_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
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


# kernel path: /tmp/torchinductor_youkaichao/qh/cqhplkdmir3k2myv4y6rl5cb3ldlpzuiprncuxy5l5qqk3qgbs7i.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((16*(((r2 + (128*x1)) // 16) % 16)) + (256*x0) + (131072*((r2 + (128*x1)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ol/colvbfxvw75fwlgw6x2lzhiiyagwap56e6n3rzgevraggqve4ieh.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_leaky_relu_backward_native_batch_norm_backward_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_leaky_relu_backward_native_batch_norm_backward_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wd/cwdd7xwzr7ctup2ivzeesalxcsbhedxtjhj2c4omeung2ntu5rki.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (x2 + (512*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (256*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x2 + (512*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.01
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp2, tmp3, tmp5)
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
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aw/caw663d35k3qas5fz3xzsw4epbk44wufcswjndzxqa7uyhjc62ld.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
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
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (32768*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (65536 + (16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (131072*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sc/cscifyyypomu32cfdc2ynp22ic7ab7v3y5c2yydaukn63rtoxy4f.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_leaky_relu_backward_native_batch_norm_backward_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_leaky_relu_backward_native_batch_norm_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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


# kernel path: /tmp/torchinductor_youkaichao/2n/c2nrcr7v67l7xymk5bzwdesr4fkunyszbprixdvn5np4nipnb4xx.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (65536 + (16*(((r2 + (128*x1)) // 16) % 16)) + (256*x0) + (131072*((r2 + (128*x1)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d6/cd6uhuiekb2tig6y4l3q4jvddt6qo2s57zdow7ogpkd2x4k3xgqq.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_leaky_relu_backward_native_batch_norm_backward_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_leaky_relu_backward_native_batch_norm_backward_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 16
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/ui/cuipzwozzcwjqhhbszfxhgzspz7u6qpyiw5b5kcpcvmzjqzkwdi3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (65536 + y0 + (256*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tl.where(tmp0, tmp1, tmp3)
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
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/43/c43zteyv5auu4fr7tg5unovotcvq4jnbuoyjv533qfxd4vgaultf.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
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
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (32768*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (65536*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tc/ctcpvwuowpzhcpt2lamxkrdoxdpellykmvi5ofmhpvhf7tmxks7u.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((16*(((r2 + (128*x1)) // 16) % 16)) + (256*x0) + (65536*((r2 + (128*x1)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wd/cwdn3roze7yqxiqmnzq5owbpuhtadkyrajmwbzv2owxpm5n3jerl.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (256*x2) + (65536*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tl.where(tmp0, tmp1, tmp3)
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
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hb/chbyrmxvlnu4dxklecxuwha3d4g4ghsmfjp6bede6li5gsaqut36.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (32768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (65536*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mb/cmbjd3zgore26xvyqg755dguojddpv4zkb2bc4mvji2oj6xq4jlw.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((16*(((r2 + (128*x1)) // 16) % 16)) + (256*x0) + (65536*((r2 + (128*x1)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d5/cd5kzjv23bszwe5nmvg7fqumfw6mycu3pp7f7wgh64mzdxuhtjed.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (256*x2) + (65536*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.01
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp2, tmp3, tmp5)
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
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sc/cscdyo776xqhz2ft556vffkibvri7wqfuykyfx77th524wx3vceg.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
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
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r3)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp0, tmp3, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2n/c2nb5pbvnomib4utgrocrfllyo4bpimpg53zzgbkmsvyomi3m2a6.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
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
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (32768*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (65536*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (65536*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (256*r2) + (32768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp0, tmp3, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qm/cqm6gep3zq77y3q6rhj6666t7lkx7kcuahvjlvgewsdjip23qwa2.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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


# kernel path: /tmp/torchinductor_youkaichao/kd/ckdfyu6wwlqc4pduoj5knu73a54klo5qq7f7xyxk235gs7g2h27l.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_leaky_relu_backward_native_batch_norm_backward_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_backward_native_batch_norm_backward_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (256*x2) + (65536*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (256*x2) + (65536*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.01
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp0, tmp3, tmp5)
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
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/db/cdbyfap2sk2afeu3qchk3t22ci6h3zvdllymltouko2kr2dl4ga6.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r3)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (x0 + (256*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp5 = tmp3 + tmp4
        tmp6 = 0.01
        tmp7 = tmp5 * tmp6
        tmp8 = tl.where(tmp0, tmp5, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp8 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp17, xmask)
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr2 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5t/c5tjkyy72rm7qhhinrps3vktt55lbxiftj4zyt2tsaucfqgyfgkv.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (65536*y1)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0 + (256*x2) + (65536*y1)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.01
    tmp7 = tmp5 * tmp6
    tmp8 = tl.where(tmp0, tmp5, tmp7)
    tmp11 = tmp9 - tmp10
    tmp13 = 0.00048828125
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tl.store(out_ptr1 + (y0 + (256*x2) + (65536*y1)), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7b/c7bhmklm45haspgfpypbeozf4pexdypus6s3uflknpx5xeafwauj.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward]

triton_poi_fused_add_leaky_relu_backward_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_backward_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (65536*y1)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.01
    tmp9 = tmp7 * tmp8
    tmp10 = tl.where(tmp0, tmp7, tmp9)
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l6/cl636mgitnxjo56c2yavegbxeda6agnu53b54lgrm52gyknvdbwv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
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
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sn/csnqtflfilnjg6s3kksf2syrlnw5k6dfykwiz4gipueyn4pc3ao3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
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
        tmp0 = tl.load(in_ptr0 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (65536*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (256*r2) + (32768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dd/cddhcxokuzpheiv4nswi5kbf5tc3cjcpuav7vgmi4dxqj6mg5nad.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (65536*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cg/ccg3czwwxtckpahqxnpu6ctzzvpnkaud5mweesnziie66hevcoyz.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_53', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fv/cfvlgd5sapulpdjxyopw6cjafvuthp7kmfcvtu43p4xcezrd5dct.py
# Source Nodes: [], Original ATen: [aten.cat]

triton_poi_fused_cat_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_54', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256) % 512
    x3 = xindex
    x2 = (xindex // 131072)
    x4 = xindex % 131072
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_out_ptr0 + (x3), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 512, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr0 + ((-65536) + x4 + (65536*x2)), tmp8, other=0.0)
    tmp12 = tl.load(in_ptr1 + ((-65536) + x4 + (65536*x2)), tmp8, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr2 + ((-65536) + x4 + (65536*x2)), tmp8, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.load(in_ptr3 + ((-65536) + x4 + (65536*x2)), tmp8, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr4 + ((-65536) + x4 + (65536*x2)), tmp8, other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp8, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(in_out_ptr0 + (x3), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rs/crs7ml564c7fxznnoth23kzzqhbwlxweymnamzclxufo3hu5zwkx.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
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
        tmp0 = tl.load(in_ptr0 + (x1 + (512*r2) + (65536*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (131072*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gt/cgtmmwiarepiaoaxiayg5zpbwjilnth4p2lsitjdncf2v64oifr6.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((16*(((r2 + (128*x1)) // 16) % 16)) + (256*x0) + (131072*((r2 + (128*x1)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i6/ci63z2xgnukzsjftmq6zc262aldcsxf3x64ihejjwemz4k7zbudd.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (x2 + (512*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (256*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (512*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tl.where(tmp0, tmp1, tmp3)
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
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iq/ciqttoayz3baurch4gvvu7mt54ogwyr3g37x2ajibmpdfv6hkhdk.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (32768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (262144*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ro/croc563pbe7jfxnljc2twle55bkyyysdchxwrla2yskznhsaszdz.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_leaky_relu_backward_native_batch_norm_backward_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_leaky_relu_backward_native_batch_norm_backward_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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


# kernel path: /tmp/torchinductor_youkaichao/vu/cvu5u4adufzaqsdx5bspot5bupit3d2yst7qghl746oiya4jgzuc.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((32*(((r2 + (128*x1)) // 32) % 32)) + (1024*x0) + (262144*((r2 + (128*x1)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h7/ch7hwzscp3l7coef657zbs7cwhhlar3von6xxhf3rndrdko5yjbr.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_leaky_relu_backward_native_batch_norm_backward_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_leaky_relu_backward_native_batch_norm_backward_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 64
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/xm/cxmgvhivjnodgvpkna2hpnkvdm3yaowjilrpp7tsq5z3bhvc4b2y.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_62', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (1024*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.01
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp2, tmp3, tmp5)
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
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4u/c4uyngxtup4inab5e42s3gyflfygvunv3ksooufez2mz6fui6kfn.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
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
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (131072 + (32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (262144*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/y2/cy2go5pqq6si4q5m52jomaty4mifmlctxuzuqajo3a7r3abotjk7.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_leaky_relu_backward_native_batch_norm_backward_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_leaky_relu_backward_native_batch_norm_backward_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
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


# kernel path: /tmp/torchinductor_youkaichao/uj/cujwm4yo3sdr7qdnincsde7jiwt2cd7dsmdxnnxwqtdb7aw3xo6p.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (131072 + (32*(((r2 + (128*x1)) // 32) % 32)) + (1024*x0) + (262144*((r2 + (128*x1)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yt/cytudvx45j7w6otjtzcydd7m2sdt2qft33bc5rdfj2ziovmglgpr.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_leaky_relu_backward_native_batch_norm_backward_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_leaky_relu_backward_native_batch_norm_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 64
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/um/cum33rgwzp7w4x2tpduteddetoeee5iel3ywqrftcj5x6zhpgcfu.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (131072 + y0 + (1024*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tl.where(tmp0, tmp1, tmp3)
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
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nf/cnffaotlssmuxlc6uotmpqkqcel4ofzkfgdsvvx6xf4gsizfi4jp.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
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
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (131072*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ro/croufvj5zxnh2nhh4kzghkcgjmnpquf3dhwoji3tshn5rffqnk67.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((32*(((r2 + (128*x1)) // 32) % 32)) + (1024*x0) + (131072*((r2 + (128*x1)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fs/cfs33loxptr2nyaj2zwhaq4pbckme47l64tp2zvixej75yvlssbo.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (1024*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tl.where(tmp0, tmp1, tmp3)
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
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7y/c7yg5ufzjfwpzif27ghcszhohhlpwe23zahg5xnzckwvaotr7r2z.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (131072*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xg/cxgepczkjwrk2wjea7topvfvqsdhkdczxbtae3fss4jj2tn2xfmv.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((32*(((r2 + (128*x1)) // 32) % 32)) + (1024*x0) + (131072*((r2 + (128*x1)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dk/cdke3faae6qem4kbdgjxj36zcldrbdhy24b37wforuvmx2euclio.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (1024*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.01
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp2, tmp3, tmp5)
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
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xd/cxd7pmbppkystpl5y3srswcxqrxnckvbkfy43drlq5kuf4vonyzu.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_74 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r3)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp0, tmp3, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bz/cbzq5vcqiuy7rjgbjtq2l3t47p743cvgwd42ews4w32oz5y2qqxy.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_75 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
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
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (131072*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (131072*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (128*r2) + (16384*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp0, tmp3, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ey/ceysk43bbe246ooqwuifxh2g6yc6otbjm63iyphmxy3fmex22l5t.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
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


# kernel path: /tmp/torchinductor_youkaichao/kb/ckb3m74hdfksddromrcepprlu7d4p3dq6wv7nredhfo6wpkcauoh.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_leaky_relu_backward_native_batch_norm_backward_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_backward_native_batch_norm_backward_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (1024*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (1024*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.01
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp0, tmp3, tmp5)
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
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/js/cjsgh7y7xoajhh7wygafwvf2yl4ky4yh3fcs4bzw7ji2xlag5pym.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r3)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (x0 + (128*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp5 = tmp3 + tmp4
        tmp6 = 0.01
        tmp7 = tmp5 * tmp6
        tmp8 = tl.where(tmp0, tmp5, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp8 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp17, xmask)
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr2 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pm/cpm6aplmzffonujl6tdto2seoowquu5v3mypu76jn6giotboono5.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (131072*y1)), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2 + (1024*y3)), xmask)
    tmp2 = tl.load(in_ptr2 + (x2 + (1024*y3)), xmask)
    tmp4 = tl.load(in_ptr3 + (x2 + (1024*y3)), xmask)
    tmp9 = tl.load(in_ptr4 + (y0 + (128*x2) + (131072*y1)), xmask)
    tmp10 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.01
    tmp7 = tmp5 * tmp6
    tmp8 = tl.where(tmp0, tmp5, tmp7)
    tmp11 = tmp9 - tmp10
    tmp13 = 0.0001220703125
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tl.store(out_ptr1 + (y0 + (128*x2) + (131072*y1)), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jn/cjnfuji6u22u3scdlm3hg3kp2y5mpzji7izford4ntvgqdtrn3kr.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward]

triton_poi_fused_add_leaky_relu_backward_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_backward_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (131072*y1)), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2 + (1024*y3)), xmask)
    tmp2 = tl.load(in_ptr2 + (x2 + (1024*y3)), xmask)
    tmp4 = tl.load(in_ptr3 + (x2 + (1024*y3)), xmask)
    tmp6 = tl.load(in_ptr4 + (x2 + (1024*y3)), xmask)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.01
    tmp9 = tmp7 * tmp8
    tmp10 = tl.where(tmp0, tmp7, tmp9)
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/or/corhh3ejnlufndvlvjazxir5b4azkpga3geo42b6agy3dmjl5ole.py
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
    xnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lc/clcqigeiyianwcpec7y22px2egpgnwesyd5zol5cvpfaegjzyseb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_82', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
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
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (131072*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (128*r2) + (16384*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/iy/ciytur4em2maqmhia7q2kdhyjp4cvdzj56rwwymreinvaxzj5g7b.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_83 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_83', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m5/cm52s6ccf6vnq7s53asrr4hxaakqpw5wxlcqvv5ichq7rwxdiqy7.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_84', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kb/ckb2ehme6rynbnac3w6goaig7d6fdgb6oi7vlsw3xmo2uqalhuhp.py
# Source Nodes: [], Original ATen: [aten.cat]

triton_poi_fused_cat_85 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_85', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 1024) % 256
    x3 = xindex
    x2 = (xindex // 262144)
    x4 = xindex % 262144
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_out_ptr0 + (x3), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 256, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr0 + ((-131072) + x4 + (131072*x2)), tmp8, other=0.0)
    tmp12 = tl.load(in_ptr1 + ((-131072) + x4 + (131072*x2)), tmp8, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr2 + ((-131072) + x4 + (131072*x2)), tmp8, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.load(in_ptr3 + ((-131072) + x4 + (131072*x2)), tmp8, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr4 + ((-131072) + x4 + (131072*x2)), tmp8, other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp8, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(in_out_ptr0 + (x3), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3p/c3p6xtjqgt4hovsv3zwaza63o4okwtyry3pkioyaod76ocou3g62.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_86', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
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
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (32768*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (262144*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hd/chd77b4jtql4aybxvunvkgbhnnorsh25scndx5jfdve3vhg3zfvj.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((32*(((r2 + (128*x1)) // 32) % 32)) + (1024*x0) + (262144*((r2 + (128*x1)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/do/cdogyuitc5bnfa57ur3thk2qale3ezlltaxsp57cnc2g54dz43d3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (1024*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tl.where(tmp0, tmp1, tmp3)
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
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lr/clrete4i5j7jh24ida7mq7al44kj7floxxlgh3qvugbkk4ukinzx.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_89', 'mutated_arg_names': []}
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
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (524288*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u2/cu23suu7fnqm4ajlay2kimo6qhgpmv4qzupox7b2hmp23ibd65bs.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_leaky_relu_backward_native_batch_norm_backward_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_leaky_relu_backward_native_batch_norm_backward_90', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/j5/cj5f3662ttkrww3fhd4sxqdcpwju4imveyzkquuiyqlora6nc7vh.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_91', 'mutated_arg_names': []}
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
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((64*(((r2 + (128*x1)) // 64) % 64)) + (4096*x0) + (524288*((r2 + (128*x1)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wx/cwxemlhg37o6eurvfn4ra5ntvn7v6i566usf4icagvcwv3w4vreo.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_92', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ap/capacfuygvtmdpoz2ceefz23oe35cgxo2zsl7evlmupjtnws6p2x.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_93', 'mutated_arg_names': []},
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
    tmp7 = tl.load(in_ptr2 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.01
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp2, tmp3, tmp5)
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
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mi/cmi7jidlpawztctnu7z743fzoav6isrot5ma7yzfzioh6fvepxu7.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_94 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_94', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
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
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (262144 + (64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (524288*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dz/cdzykgf2bljnzsm6ywe72c2i5zbh7khqwyb3jcpj4i5lah3l2m6h.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_leaky_relu_backward_native_batch_norm_backward_95 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_leaky_relu_backward_native_batch_norm_backward_95', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 64
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


# kernel path: /tmp/torchinductor_youkaichao/54/c54e4oubwujptfxcm4m6dmhadcaq6pgem5n6cjxwluk2d53eapzc.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_96 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_96', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (262144 + (64*(((r2 + (128*x1)) // 64) % 64)) + (4096*x0) + (524288*((r2 + (128*x1)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h7/ch7gvl4lvlcnju3x3zpao6v42vkkpxgm7ek35qk5viy6wdeerepi.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_97 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_97', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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


# kernel path: /tmp/torchinductor_youkaichao/zz/czzf4hq2cahing6bafsvtj7yewexclwzr6hq6l3imcfye4vghwc4.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_98 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_98', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (262144 + y0 + (4096*x2) + (524288*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tl.where(tmp0, tmp1, tmp3)
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
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ix/cix6xg5oz7uliglaljr43yz4c44kpvgicsd4twp3jbj7c2bvfgcd.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_99', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
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
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (262144*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jt/cjtbpqfricsdrwb5r5iv3jlvanh2rjx7kzyz2d63hui3wh3q7rth.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_100 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_100', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((64*(((r2 + (128*x1)) // 64) % 64)) + (4096*x0) + (262144*((r2 + (128*x1)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/op/copm2r4q43nlqbmjoexcn274mpuwkddwj2fif5tksuzozxb6pjl6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_101 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_101', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (4096*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tl.where(tmp0, tmp1, tmp3)
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
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pe/cpeqemnojfgjwfdkklh4updcjdjbjryd3nw7kodb6wzvtyjxaphd.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_102 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_102', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (262144*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/er/cerbqb2dtamek7y3k4s2dxsmniz3dulejib25fkrqmedzhipijtl.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_103 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_103', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((64*(((r2 + (128*x1)) // 64) % 64)) + (4096*x0) + (262144*((r2 + (128*x1)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sn/csn7qwy5bgsejatwcrlir2zetr33j7l3v75vr2igi55ckuz27t3x.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_104 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_104', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (4096*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.01
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp2, tmp3, tmp5)
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
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zq/czqbhzoxt235e45phf5d4el2ozmm5loecrr3wq7bulyh6wqziy7g.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_105 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_105', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (524288*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp0, tmp3, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pk/cpkevwrj4t4fdsv2brzep5jhuozahpcxzx5xvzjbqbm7m44vgarq.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_106 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_106', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/sp/cspmvbzgzckv4jmmgaeubcse3glzm6h257xmf7y3w2n5tm3zgim5.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_107 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_107', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
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
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (262144*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (262144*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp0, tmp3, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r4/cr4yyr6un2dnu4idvy7z3laf63wiycjhiz6deub3onzkr6ofxftm.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_108 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_108', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 64
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


# kernel path: /tmp/torchinductor_youkaichao/hp/chp6axnmzq7lnxerm5cfmybu24qcn4ziepm3i3prcgfownipoipw.py
# Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_leaky_relu_backward_native_batch_norm_backward_109 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_backward_native_batch_norm_backward_109', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (4096*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (4096*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.01
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp0, tmp3, tmp5)
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
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ex/cexa4r46qsmzsptntj4apw5lga4lwzhn6glxa53w76jd4d7xwjet.py
# Source Nodes: [], Original ATen: [aten.cat, aten.leaky_relu_backward]

triton_poi_fused_cat_leaky_relu_backward_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_leaky_relu_backward_110', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (524288*y1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 64, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_out_ptr0 + (x2 + (4096*y3)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 128, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr1 + ((-262144) + x2 + (4096*y0) + (262144*y1)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr2 + ((-262144) + x2 + (4096*y0) + (262144*y1)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.load(in_ptr3 + ((-262144) + x2 + (4096*y0) + (262144*y1)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp9, tmp16, tmp17)
    tmp19 = tl.where(tmp5, tmp8, tmp18)
    tmp20 = 0.01
    tmp21 = tmp19 * tmp20
    tmp22 = tl.where(tmp0, tmp19, tmp21)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (4096*y3)), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/p7/cp7u4fvu3ummoskpttlrmwmmxl5edbv5v5juac6uh5sxygmlimbl.py
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
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_111', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6x/c6xmdlqbxyzxqqoobpshzsyl7il26sb42virjyidmk74hi3z4vw3.py
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
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_112', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/sc/cscsucictsdldfdxvhg6epu7ha6tcbi7wdijacdik26iddsqrnzr.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_113', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (524288*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (128*r2) + (16384*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yh/cyhca3gvblov6umfpq7blkyk4y6kprswrgl2e4qw4emylgmjwvm5.py
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
    size_hints=[128, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_114', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/i6/ci6enktelholxzpkahojnwadfup2pt4h7gb5rivlvgfdte5a3wfg.py
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
    size_hints=[32768, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_115', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 4096
    y1 = (yindex // 4096)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (4096*x2) + (524288*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 3.0517578125e-05
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/22/c22gcsqdy4tavqsmxheozvil52litisvyrkelqr6uy7d2moi2kgm.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_116 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_116', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r2 + (128*(x0 % 128)) + (16384*x1) + (1048576*((r2 + (128*x0)) // 16384))), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ie/cieaue3oxyi6xuumsj26p2jrv6zmmekgfiwvzarta5koaeygddrn.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_leaky_relu_backward_native_batch_norm_backward_117 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_leaky_relu_backward_native_batch_norm_backward_117', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 64
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
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oq/coqndei4mk7zjdhsugl54up35mippeahct7aq37gn733fkqz3yuj.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_118 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_118', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r2 + (128*(x1 % 128)) + (16384*x0) + (1048576*((r2 + (128*x1)) // 16384))), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ab/cabckwelp6kljaeupe33j5kipnme2x6ocs6znztcy6nafd3aa4rp.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_119 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_119', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 1024
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


# kernel path: /tmp/torchinductor_youkaichao/5s/c5sppdwsp5srudimrr36vsdwbeqkerhxycf2kskmzkxcutu2hjjb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_120 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_120', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (16384*x2) + (1048576*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.01
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp2, tmp3, tmp5)
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
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ht/chtf4flxd3yhpp7amrl4zhrqyywqiqggqmefkuhgwpb4hsdaxg62.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_121 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_121', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (1048576 + r2 + (128*(x0 % 128)) + (16384*x1) + (2097152*((r2 + (128*x0)) // 16384))), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l5/cl5hemgn2424kogku34l3ttkr2jrsw7ptzfxnjc453oiiy3sgr5g.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_122 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_122', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (1048576 + r2 + (128*(x1 % 128)) + (16384*x0) + (2097152*((r2 + (128*x1)) // 16384))), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ph/cphybjgqdb5hmdymiwb7uqonewkftigvjythhm6a3hm6bpfudhtm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_123 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_123', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (1048576 + y0 + (16384*x2) + (2097152*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tl.where(tmp0, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 7.62939453125e-06
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


# kernel path: /tmp/torchinductor_youkaichao/dw/cdwqveiztgs5me23bwkvjxntowaoikkbb6jk375qochly4ijwsz5.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_124 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_124', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r2 + (128*(x0 % 128)) + (16384*x1) + (1048576*((r2 + (128*x0)) // 16384))), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u7/cu7kbqau52syhmxtnufvgw6suwugnjneko5fpj3px5q6etz7lpam.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_125 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_125', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r2 + (128*(x1 % 128)) + (16384*x0) + (1048576*((r2 + (128*x1)) // 16384))), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.01
        tmp3 = tmp1 * tmp2
        tmp4 = tl.where(tmp0, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bx/cbx52pqk4cuflvqa464dkb6stof4dfjpfimqwtvg2xotqgepun77.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_126 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_126', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (16384*x2) + (1048576*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tl.where(tmp0, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 7.62939453125e-06
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


# kernel path: /tmp/torchinductor_youkaichao/ao/caorl6ocldjyex5isabm74namhdmcesdok25jagnyxhtwz4edvam.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_127 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_127', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (32*r2) + (4096*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r2 + (128*(x0 % 128)) + (16384*x1) + (524288*((r2 + (128*x0)) // 16384))), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hl/chltag45xju3xd4ogvooggr6drhvr4odvbi6zfwg3v7qoqmx7a32.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_leaky_relu_backward_native_batch_norm_backward_128 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_leaky_relu_backward_native_batch_norm_backward_128', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
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
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7g/c7glrgfhc3u3l6hajfcirshps2znt3gl222dgnukonecajzv7liq.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_129 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_129', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (4096*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r2 + (128*(x1 % 128)) + (16384*x0) + (524288*((r2 + (128*x1)) // 16384))), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (32*r2) + (4096*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ke/ckexli5hockjcwl6qpwh75a735tilbsayy7motpwujkm2dhk3hd6.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_130 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_130', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 1024
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


# kernel path: /tmp/torchinductor_youkaichao/cg/ccg2hoqpkxnjipinpvhkfikou7j3zdvptgch4i74j7pcvek7vsda.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_131 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_131', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp7 = tl.load(in_ptr2 + (x2 + (32*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.01
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp2, tmp3, tmp5)
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


# kernel path: /tmp/torchinductor_youkaichao/ai/caijdjutqkyrr2vkwwbewli2zn5zbmmollhodghqpkn7y2sio5df.py
# Source Nodes: [], Original ATen: [aten.cat, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_cat_leaky_relu_backward_native_batch_norm_backward_132 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_leaky_relu_backward_native_batch_norm_backward_132', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    x3 = xindex
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp25 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (4194304*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp24 = tl.load(in_ptr4 + (x0 + (128*r2) + (4194304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = x0
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1, 1], 64, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = tl.load(in_ptr1 + ((16384*x0) + (2097152*(r2 // 16384)) + (4194304*x1) + (r2 % 16384)), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tmp1 >= tmp4
        tmp10 = tl.full([1, 1], 128, tl.int64)
        tmp11 = tmp1 < tmp10
        tmp12 = tl.load(in_ptr2 + ((-1048576) + (16384*x3) + (1048576*(r2 // 16384)) + (r2 % 16384)), rmask & tmp9 & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr3 + ((-1048576) + (16384*x3) + (1048576*(r2 // 16384)) + (r2 % 16384)), rmask & tmp9 & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tmp12 + tmp13
        tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
        tmp16 = tl.where(tmp9, tmp14, tmp15)
        tmp17 = tl.where(tmp5, tmp8, tmp16)
        tmp18 = 0.01
        tmp19 = tmp17 * tmp18
        tmp20 = tl.where(tmp0, tmp17, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
        tmp26 = tmp24 - tmp25
        tmp27 = tmp20 * tmp26
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask & xmask, tmp30, _tmp29)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, xmask)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lc/clcnq3luuhsfizsenuekceeeje4fucc32bx3u724by2yx6rrv2bz.py
# Source Nodes: [], Original ATen: [aten.cat, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_per_fused_cat_leaky_relu_backward_native_batch_norm_backward_133 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_leaky_relu_backward_native_batch_norm_backward_133', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/yg/cyg2bnqy5pdkcendiqjkbv72jvr66jsa2c2rj6tffcxiv3rdxsm2.py
# Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_cat_convolution_backward_leaky_relu_backward_native_batch_norm_backward_134 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_backward_leaky_relu_backward_native_batch_norm_backward_134', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (2097152*y1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp21 = tl.load(in_ptr3 + (y0 + (128*x2) + (2097152*y1)), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 64, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_out_ptr0 + (x2 + (16384*y3)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 128, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr1 + ((-1048576) + x2 + (16384*y0) + (1048576*y1)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr2 + ((-1048576) + x2 + (16384*y0) + (1048576*y1)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp9, tmp14, tmp15)
    tmp17 = tl.where(tmp5, tmp8, tmp16)
    tmp18 = 0.01
    tmp19 = tmp17 * tmp18
    tmp20 = tl.where(tmp0, tmp17, tmp19)
    tmp23 = tmp21 - tmp22
    tmp25 = 7.62939453125e-06
    tmp26 = tmp24 * tmp25
    tmp28 = tmp27 * tmp27
    tmp29 = tmp26 * tmp28
    tmp30 = tmp23 * tmp29
    tmp31 = tmp20 - tmp30
    tmp33 = tmp32 * tmp25
    tmp34 = tmp31 - tmp33
    tmp36 = tmp27 * tmp35
    tmp37 = tmp34 * tmp36
    tl.store(out_ptr0 + (y0 + (128*x2) + (2097152*y1)), tmp37, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wy/cwyfbuiivhu6vzo2emqhhryoo7pdtrfftmuojfp3pbcv4kizb3uj.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_135 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_135', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (32*r2) + (16384*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((256*(((r2 + (512*x0)) // 256) % 256)) + (65536*x1) + (2097152*((r2 + (512*x0)) // 65536)) + (r2 % 256)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kc/ckcsj633xr5prb3j3d42nm4zwekq3xr7fhwmh735zz7ivuwxru4l.py
# Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_red_fused_leaky_relu_backward_native_batch_norm_backward_136 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_136', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((256*(((r2 + (512*x1)) // 256) % 256)) + (65536*x0) + (2097152*((r2 + (512*x1)) // 65536)) + (r2 % 256)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (32*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp2, tmp3, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/og/cog42gw3hczieofpjwryq5yvc5tysvdz353emzhr642o42wupqvo.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_137 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_137', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 65536
    y1 = (yindex // 65536)
    tmp0 = tl.load(in_ptr0 + (x2 + (32*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (65536*x2) + (2097152*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x2 + (32*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.01
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp2, tmp3, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 1.9073486328125e-06
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
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_405, convolution, squeeze_1, where, convolution_1, squeeze_4, where_1, convolution_2, squeeze_7, getitem_9, convolution_3, squeeze_10, where_3, convolution_4, squeeze_13, add_25, convolution_5, squeeze_16, cat, convolution_6, squeeze_19, where_6, convolution_7, squeeze_22, where_7, convolution_8, squeeze_25, getitem_27, convolution_9, squeeze_28, where_9, convolution_10, squeeze_31, add_56, convolution_11, squeeze_34, where_11, convolution_12, squeeze_37, add_67, convolution_13, squeeze_40, cat_1, convolution_14, squeeze_43, where_14, convolution_15, squeeze_46, where_15, convolution_16, squeeze_49, getitem_49, convolution_17, squeeze_52, where_17, convolution_18, squeeze_55, add_98, convolution_19, squeeze_58, where_19, convolution_20, squeeze_61, add_109, convolution_21, squeeze_64, where_21, convolution_22, squeeze_67, add_120, convolution_23, squeeze_70, where_23, convolution_24, squeeze_73, add_131, convolution_25, squeeze_76, where_25, convolution_26, squeeze_79, add_142, convolution_27, squeeze_82, where_27, convolution_28, squeeze_85, add_153, convolution_29, squeeze_88, where_29, convolution_30, squeeze_91, add_164, convolution_31, squeeze_94, where_31, convolution_32, squeeze_97, add_175, convolution_33, squeeze_100, cat_2, convolution_34, squeeze_103, where_34, convolution_35, squeeze_106, where_35, convolution_36, squeeze_109, getitem_95, convolution_37, squeeze_112, where_37, convolution_38, squeeze_115, add_206, convolution_39, squeeze_118, where_39, convolution_40, squeeze_121, add_217, convolution_41, squeeze_124, where_41, convolution_42, squeeze_127, add_228, convolution_43, squeeze_130, where_43, convolution_44, squeeze_133, add_239, convolution_45, squeeze_136, where_45, convolution_46, squeeze_139, add_250, convolution_47, squeeze_142, where_47, convolution_48, squeeze_145, add_261, convolution_49, squeeze_148, where_49, convolution_50, squeeze_151, add_272, convolution_51, squeeze_154, where_51, convolution_52, squeeze_157, add_283, convolution_53, squeeze_160, cat_3, convolution_54, squeeze_163, where_54, convolution_55, squeeze_166, where_55, convolution_56, squeeze_169, getitem_141, convolution_57, squeeze_172, where_57, convolution_58, squeeze_175, add_314, convolution_59, squeeze_178, where_59, convolution_60, squeeze_181, add_325, convolution_61, squeeze_184, where_61, convolution_62, squeeze_187, add_336, convolution_63, squeeze_190, where_63, convolution_64, squeeze_193, add_347, convolution_65, squeeze_196, cat_4, convolution_66, squeeze_199, clone, permute_1, gt_67, unsqueeze_270, gt_68, unsqueeze_282, gt_69, unsqueeze_294, unsqueeze_306, gt_71, unsqueeze_318, unsqueeze_330, gt_73, unsqueeze_342, unsqueeze_354, gt_75, unsqueeze_366, unsqueeze_378, gt_77, unsqueeze_390, unsqueeze_402, unsqueeze_414, gt_80, unsqueeze_426, gt_81, unsqueeze_438, unsqueeze_450, gt_83, unsqueeze_462, unsqueeze_474, gt_85, unsqueeze_486, unsqueeze_498, gt_87, unsqueeze_510, unsqueeze_522, gt_89, unsqueeze_534, unsqueeze_546, gt_91, unsqueeze_558, unsqueeze_570, gt_93, unsqueeze_582, unsqueeze_594, gt_95, unsqueeze_606, unsqueeze_618, gt_97, unsqueeze_630, unsqueeze_642, unsqueeze_654, gt_100, unsqueeze_666, gt_101, unsqueeze_678, unsqueeze_690, gt_103, unsqueeze_702, unsqueeze_714, gt_105, unsqueeze_726, unsqueeze_738, gt_107, unsqueeze_750, unsqueeze_762, gt_109, unsqueeze_774, unsqueeze_786, gt_111, unsqueeze_798, unsqueeze_810, gt_113, unsqueeze_822, unsqueeze_834, gt_115, unsqueeze_846, unsqueeze_858, gt_117, unsqueeze_870, unsqueeze_882, unsqueeze_894, gt_120, unsqueeze_906, gt_121, unsqueeze_918, unsqueeze_930, gt_123, unsqueeze_942, unsqueeze_954, gt_125, unsqueeze_966, unsqueeze_978, unsqueeze_990, gt_128, unsqueeze_1002, gt_129, unsqueeze_1014, unsqueeze_1026, gt_131, unsqueeze_1038, unsqueeze_1050, unsqueeze_1062, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_31, (256, ), (1, ))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_53, (128, ), (1, ))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_57, (128, ), (1, ))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_61, (128, ), (1, ))
    assert_size_stride(primals_63, (128, ), (1, ))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_67, (128, ), (1, ))
    assert_size_stride(primals_69, (256, ), (1, ))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_73, (512, ), (1, ))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_77, (256, ), (1, ))
    assert_size_stride(primals_79, (256, ), (1, ))
    assert_size_stride(primals_81, (256, ), (1, ))
    assert_size_stride(primals_83, (256, ), (1, ))
    assert_size_stride(primals_85, (256, ), (1, ))
    assert_size_stride(primals_87, (256, ), (1, ))
    assert_size_stride(primals_89, (256, ), (1, ))
    assert_size_stride(primals_91, (256, ), (1, ))
    assert_size_stride(primals_93, (256, ), (1, ))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_97, (256, ), (1, ))
    assert_size_stride(primals_99, (256, ), (1, ))
    assert_size_stride(primals_101, (256, ), (1, ))
    assert_size_stride(primals_103, (256, ), (1, ))
    assert_size_stride(primals_105, (256, ), (1, ))
    assert_size_stride(primals_107, (256, ), (1, ))
    assert_size_stride(primals_109, (512, ), (1, ))
    assert_size_stride(primals_111, (1024, ), (1, ))
    assert_size_stride(primals_113, (1024, ), (1, ))
    assert_size_stride(primals_115, (512, ), (1, ))
    assert_size_stride(primals_117, (512, ), (1, ))
    assert_size_stride(primals_119, (512, ), (1, ))
    assert_size_stride(primals_121, (512, ), (1, ))
    assert_size_stride(primals_123, (512, ), (1, ))
    assert_size_stride(primals_125, (512, ), (1, ))
    assert_size_stride(primals_127, (512, ), (1, ))
    assert_size_stride(primals_129, (512, ), (1, ))
    assert_size_stride(primals_131, (512, ), (1, ))
    assert_size_stride(primals_133, (1024, ), (1, ))
    assert_size_stride(primals_135, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_136, (64, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_137, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_138, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_139, (64, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_140, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_141, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_142, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_143, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_144, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_145, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_146, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_147, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_148, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_149, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_150, (256, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_151, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_152, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_153, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_154, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_155, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_156, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_157, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_158, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_159, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_160, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_161, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_162, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_163, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_164, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_165, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_166, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_167, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_168, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_169, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_170, (512, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_171, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_172, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_173, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_174, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_175, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_176, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_177, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_178, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_179, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_180, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_181, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_182, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_183, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_184, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_185, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_186, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_187, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_188, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_189, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_190, (1024, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_191, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_192, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_193, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_194, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_195, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_196, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_197, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_198, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_199, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_200, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_201, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_405, (8, 3, 256, 256), (196608, 1, 768, 3))
    assert_size_stride(convolution, (8, 32, 256, 256), (2097152, 1, 8192, 32))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(where, (8, 32, 256, 256), (2097152, 1, 8192, 32))
    assert_size_stride(convolution_1, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(squeeze_4, (64, ), (1, ))
    assert_size_stride(where_1, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(convolution_2, (8, 128, 128, 128), (2097152, 1, 16384, 128))
    assert_size_stride(squeeze_7, (128, ), (1, ))
    assert_size_stride(getitem_9, (8, 64, 128, 128), (2097152, 16384, 128, 1))
    assert_size_stride(convolution_3, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(squeeze_10, (32, ), (1, ))
    assert_size_stride(where_3, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(convolution_4, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(squeeze_13, (64, ), (1, ))
    assert_size_stride(add_25, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(convolution_5, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(squeeze_16, (64, ), (1, ))
    assert_size_stride(cat, (8, 128, 128, 128), (2097152, 1, 16384, 128))
    assert_size_stride(convolution_6, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(squeeze_19, (64, ), (1, ))
    assert_size_stride(where_6, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(convolution_7, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(squeeze_22, (128, ), (1, ))
    assert_size_stride(where_7, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(convolution_8, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(squeeze_25, (128, ), (1, ))
    assert_size_stride(getitem_27, (8, 64, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(convolution_9, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_28, (64, ), (1, ))
    assert_size_stride(where_9, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_10, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_31, (64, ), (1, ))
    assert_size_stride(add_56, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_11, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_34, (64, ), (1, ))
    assert_size_stride(where_11, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_12, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_37, (64, ), (1, ))
    assert_size_stride(add_67, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_13, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_40, (64, ), (1, ))
    assert_size_stride(cat_1, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(convolution_14, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(squeeze_43, (128, ), (1, ))
    assert_size_stride(where_14, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(convolution_15, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(squeeze_46, (256, ), (1, ))
    assert_size_stride(where_15, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(convolution_16, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(squeeze_49, (256, ), (1, ))
    assert_size_stride(getitem_49, (8, 128, 32, 32), (262144, 1024, 32, 1))
    assert_size_stride(convolution_17, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_52, (128, ), (1, ))
    assert_size_stride(where_17, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_18, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_55, (128, ), (1, ))
    assert_size_stride(add_98, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_19, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_58, (128, ), (1, ))
    assert_size_stride(where_19, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_20, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_61, (128, ), (1, ))
    assert_size_stride(add_109, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_21, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_64, (128, ), (1, ))
    assert_size_stride(where_21, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_22, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_67, (128, ), (1, ))
    assert_size_stride(add_120, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_23, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_70, (128, ), (1, ))
    assert_size_stride(where_23, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_24, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_73, (128, ), (1, ))
    assert_size_stride(add_131, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_25, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_76, (128, ), (1, ))
    assert_size_stride(where_25, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_26, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_79, (128, ), (1, ))
    assert_size_stride(add_142, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_27, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_82, (128, ), (1, ))
    assert_size_stride(where_27, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_28, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_85, (128, ), (1, ))
    assert_size_stride(add_153, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_29, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_88, (128, ), (1, ))
    assert_size_stride(where_29, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_30, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_91, (128, ), (1, ))
    assert_size_stride(add_164, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_31, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_94, (128, ), (1, ))
    assert_size_stride(where_31, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_32, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_97, (128, ), (1, ))
    assert_size_stride(add_175, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_33, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_100, (128, ), (1, ))
    assert_size_stride(cat_2, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(convolution_34, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(squeeze_103, (256, ), (1, ))
    assert_size_stride(where_34, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(convolution_35, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(squeeze_106, (512, ), (1, ))
    assert_size_stride(where_35, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(convolution_36, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(squeeze_109, (512, ), (1, ))
    assert_size_stride(getitem_95, (8, 256, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(convolution_37, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_112, (256, ), (1, ))
    assert_size_stride(where_37, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_38, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_115, (256, ), (1, ))
    assert_size_stride(add_206, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_39, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_118, (256, ), (1, ))
    assert_size_stride(where_39, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_40, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_121, (256, ), (1, ))
    assert_size_stride(add_217, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_41, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_124, (256, ), (1, ))
    assert_size_stride(where_41, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_42, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_127, (256, ), (1, ))
    assert_size_stride(add_228, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_43, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_130, (256, ), (1, ))
    assert_size_stride(where_43, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_44, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_133, (256, ), (1, ))
    assert_size_stride(add_239, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_45, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_136, (256, ), (1, ))
    assert_size_stride(where_45, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_46, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_139, (256, ), (1, ))
    assert_size_stride(add_250, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_47, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_142, (256, ), (1, ))
    assert_size_stride(where_47, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_48, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_145, (256, ), (1, ))
    assert_size_stride(add_261, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_49, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_148, (256, ), (1, ))
    assert_size_stride(where_49, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_50, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_151, (256, ), (1, ))
    assert_size_stride(add_272, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_51, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_154, (256, ), (1, ))
    assert_size_stride(where_51, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_52, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_157, (256, ), (1, ))
    assert_size_stride(add_283, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_53, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_160, (256, ), (1, ))
    assert_size_stride(cat_3, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(convolution_54, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(squeeze_163, (512, ), (1, ))
    assert_size_stride(where_54, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(convolution_55, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
    assert_size_stride(squeeze_166, (1024, ), (1, ))
    assert_size_stride(where_55, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
    assert_size_stride(convolution_56, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
    assert_size_stride(squeeze_169, (1024, ), (1, ))
    assert_size_stride(getitem_141, (8, 512, 8, 8), (65536, 64, 8, 1))
    assert_size_stride(convolution_57, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_172, (512, ), (1, ))
    assert_size_stride(where_57, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_58, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_175, (512, ), (1, ))
    assert_size_stride(add_314, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_59, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_178, (512, ), (1, ))
    assert_size_stride(where_59, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_60, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_181, (512, ), (1, ))
    assert_size_stride(add_325, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_61, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_184, (512, ), (1, ))
    assert_size_stride(where_61, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_62, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_187, (512, ), (1, ))
    assert_size_stride(add_336, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_63, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_190, (512, ), (1, ))
    assert_size_stride(where_63, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_64, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_193, (512, ), (1, ))
    assert_size_stride(add_347, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_65, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_196, (512, ), (1, ))
    assert_size_stride(cat_4, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
    assert_size_stride(convolution_66, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
    assert_size_stride(squeeze_199, (1024, ), (1, ))
    assert_size_stride(clone, (8, 1024), (1024, 1))
    assert_size_stride(permute_1, (1000, 1024), (1024, 1))
    assert_size_stride(gt_67, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
    assert_size_stride(unsqueeze_270, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(gt_68, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_282, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(gt_69, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_294, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_306, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(gt_71, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_318, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_330, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(gt_73, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_342, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_354, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(gt_75, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_366, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_378, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(gt_77, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
    assert_size_stride(unsqueeze_390, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_414, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(gt_80, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_426, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_81, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_438, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_450, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_83, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_462, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_85, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_486, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_498, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_87, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_510, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_522, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_89, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_534, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_546, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_91, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_558, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_570, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_93, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_582, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_594, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_95, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_606, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_618, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_97, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(unsqueeze_630, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_642, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_654, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_100, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_666, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_101, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_678, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_690, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_103, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_702, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_714, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_105, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_726, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_738, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_107, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_750, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_762, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_109, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_774, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_786, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_111, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_798, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_810, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_113, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_822, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_834, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_115, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_846, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_858, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_117, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(unsqueeze_870, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_882, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_894, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_120, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(unsqueeze_906, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(gt_121, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(unsqueeze_918, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_930, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(gt_123, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(unsqueeze_942, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_954, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(gt_125, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(unsqueeze_966, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_978, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_990, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(gt_128, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(unsqueeze_1002, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(gt_129, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(unsqueeze_1014, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1026, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(gt_131, (8, 128, 128, 128), (2097152, 1, 16384, 128))
    assert_size_stride(unsqueeze_1038, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1050, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1062, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone, out=buf1)
        del clone
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty_strided((1024, 4), (1, 1024), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1024, 4), (1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_div_leaky_relu_backward_native_batch_norm_backward_1.run(gt_67, buf0, convolution_66, unsqueeze_270, buf3, buf5, 4096, 128, grid=grid(4096), stream=stream0)
        buf4 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_leaky_relu_backward_native_batch_norm_backward_2.run(buf3, buf4, 1024, 4, grid=grid(1024), stream=stream0)
        buf6 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf7 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_leaky_relu_backward_native_batch_norm_backward_3.run(buf5, squeeze_199, buf6, buf7, 1024, 4, grid=grid(1024), stream=stream0)
        buf8 = empty_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_div_leaky_relu_backward_native_batch_norm_backward_4.run(gt_67, buf0, convolution_66, unsqueeze_270, buf6, squeeze_199, buf4, primals_133, buf8, 524288, grid=grid(524288), stream=stream0)
        del convolution_66
        del gt_67
        del primals_133
        del squeeze_199
        del unsqueeze_270
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf9 = aten.convolution_backward(buf8, cat_4, primals_201, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_4
        del primals_201
        buf10 = buf9[0]
        buf11 = buf9[1]
        del buf9
        buf12 = empty_strided((512, 4), (1, 512), device='cuda', dtype=torch.float32)
        buf14 = empty_strided((512, 4), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_5.run(gt_68, buf10, convolution_65, unsqueeze_282, buf12, buf14, 2048, 128, grid=grid(2048), stream=stream0)
        buf13 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_6.run(buf12, buf13, 512, 4, grid=grid(512), stream=stream0)
        buf15 = empty((512, ), device='cuda', dtype=torch.float32)
        buf16 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_7.run(buf14, squeeze_196, buf15, buf16, 512, 4, grid=grid(512), stream=stream0)
        buf17 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_8.run(gt_68, buf10, convolution_65, unsqueeze_282, buf15, squeeze_196, buf13, primals_131, buf17, 512, 512, grid=grid(512, 512), stream=stream0)
        del convolution_65
        del gt_68
        del primals_131
        del squeeze_196
        del unsqueeze_282
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf18 = aten.convolution_backward(buf17, add_347, primals_200, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_347
        del primals_200
        buf19 = buf18[0]
        buf20 = buf18[1]
        del buf18
        buf21 = buf14; del buf14  # reuse
        buf23 = buf12; del buf12  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_9.run(gt_69, buf19, convolution_64, unsqueeze_294, buf21, buf23, 2048, 128, grid=grid(2048), stream=stream0)
        buf22 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_6.run(buf21, buf22, 512, 4, grid=grid(512), stream=stream0)
        buf24 = empty((512, ), device='cuda', dtype=torch.float32)
        buf25 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_7.run(buf23, squeeze_193, buf24, buf25, 512, 4, grid=grid(512), stream=stream0)
        buf26 = buf17; del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_10.run(gt_69, buf19, convolution_64, unsqueeze_294, buf24, squeeze_193, buf22, primals_129, buf26, 512, 512, grid=grid(512, 512), stream=stream0)
        del convolution_64
        del gt_69
        del primals_129
        del squeeze_193
        del unsqueeze_294
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf27 = aten.convolution_backward(buf26, where_63, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_199
        buf28 = buf27[0]
        buf29 = buf27[1]
        del buf27
        buf30 = buf23; del buf23  # reuse
        buf32 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_11.run(where_63, buf28, convolution_63, unsqueeze_306, buf30, buf32, 2048, 128, grid=grid(2048), stream=stream0)
        buf31 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_6.run(buf30, buf31, 512, 4, grid=grid(512), stream=stream0)
        buf33 = empty((512, ), device='cuda', dtype=torch.float32)
        buf34 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_7.run(buf32, squeeze_190, buf33, buf34, 512, 4, grid=grid(512), stream=stream0)
        buf35 = buf26; del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_12.run(where_63, buf28, convolution_63, unsqueeze_306, buf33, squeeze_190, buf31, primals_127, buf35, 512, 512, grid=grid(512, 512), stream=stream0)
        del buf28
        del convolution_63
        del primals_127
        del squeeze_190
        del unsqueeze_306
        del where_63
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf36 = aten.convolution_backward(buf35, add_336, primals_198, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_336
        del primals_198
        buf37 = buf36[0]
        buf38 = buf36[1]
        del buf36
        buf39 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_13.run(gt_71, buf19, buf37, buf39, 512, 512, grid=grid(512), stream=stream0)
        buf40 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_14.run(gt_71, buf19, buf37, convolution_62, unsqueeze_318, buf40, 2048, 128, grid=grid(2048), stream=stream0)
        buf41 = empty((512, ), device='cuda', dtype=torch.float32)
        buf43 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_7.run(buf40, squeeze_187, buf41, buf43, 512, 4, grid=grid(512), stream=stream0)
        buf42 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_leaky_relu_backward_native_batch_norm_backward_15.run(gt_71, buf19, buf37, convolution_62, unsqueeze_318, buf41, squeeze_187, buf39, primals_125, buf42, 512, 512, grid=grid(512, 512), stream=stream0)
        del convolution_62
        del gt_71
        del primals_125
        del squeeze_187
        del unsqueeze_318
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf44 = aten.convolution_backward(buf42, where_61, primals_197, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_197
        buf45 = buf44[0]
        buf46 = buf44[1]
        del buf44
        buf47 = buf40; del buf40  # reuse
        buf49 = buf30; del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_11.run(where_61, buf45, convolution_61, unsqueeze_330, buf47, buf49, 2048, 128, grid=grid(2048), stream=stream0)
        buf48 = buf41; del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_6.run(buf47, buf48, 512, 4, grid=grid(512), stream=stream0)
        buf50 = empty((512, ), device='cuda', dtype=torch.float32)
        buf51 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_7.run(buf49, squeeze_184, buf50, buf51, 512, 4, grid=grid(512), stream=stream0)
        buf52 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_12.run(where_61, buf45, convolution_61, unsqueeze_330, buf50, squeeze_184, buf48, primals_123, buf52, 512, 512, grid=grid(512, 512), stream=stream0)
        del buf45
        del convolution_61
        del primals_123
        del squeeze_184
        del unsqueeze_330
        del where_61
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf53 = aten.convolution_backward(buf52, add_325, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_325
        del primals_196
        buf54 = buf53[0]
        buf55 = buf53[1]
        del buf53
        buf56 = buf50; del buf50  # reuse
        buf57 = empty((512, ), device='cuda', dtype=torch.float32)
        buf59 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_16.run(gt_73, buf19, buf37, buf54, convolution_60, unsqueeze_342, squeeze_181, buf56, buf57, buf59, 512, 512, grid=grid(512), stream=stream0)
        buf60 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_17.run(gt_73, buf19, buf37, buf54, convolution_60, unsqueeze_342, buf57, squeeze_181, buf56, primals_121, buf60, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del convolution_60
        del gt_73
        del primals_121
        del squeeze_181
        del unsqueeze_342
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf61 = aten.convolution_backward(buf60, where_59, primals_195, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_195
        buf62 = buf61[0]
        buf63 = buf61[1]
        del buf61
        buf64 = buf49; del buf49  # reuse
        buf66 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_11.run(where_59, buf62, convolution_59, unsqueeze_354, buf64, buf66, 2048, 128, grid=grid(2048), stream=stream0)
        buf65 = buf57; del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_6.run(buf64, buf65, 512, 4, grid=grid(512), stream=stream0)
        buf67 = empty((512, ), device='cuda', dtype=torch.float32)
        buf68 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_7.run(buf66, squeeze_178, buf67, buf68, 512, 4, grid=grid(512), stream=stream0)
        buf69 = buf60; del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_12.run(where_59, buf62, convolution_59, unsqueeze_354, buf67, squeeze_178, buf65, primals_119, buf69, 512, 512, grid=grid(512, 512), stream=stream0)
        del convolution_59
        del primals_119
        del squeeze_178
        del unsqueeze_354
        del where_59
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf70 = aten.convolution_backward(buf69, add_314, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_314
        del primals_194
        buf71 = buf70[0]
        buf72 = buf70[1]
        del buf70
        buf73 = reinterpret_tensor(buf69, (8, 512, 8, 8), (32768, 64, 8, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward]
        triton_poi_fused_add_leaky_relu_backward_18.run(gt_75, buf19, buf37, buf54, buf71, buf73, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del gt_75
        buf74 = buf67; del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_19.run(buf73, buf74, 512, 512, grid=grid(512), stream=stream0)
        buf75 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_20.run(buf73, convolution_58, unsqueeze_366, buf75, 2048, 128, grid=grid(2048), stream=stream0)
        buf76 = empty((512, ), device='cuda', dtype=torch.float32)
        buf77 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_7.run(buf75, squeeze_175, buf76, buf77, 512, 4, grid=grid(512), stream=stream0)
        buf78 = reinterpret_tensor(buf62, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_21.run(buf73, convolution_58, unsqueeze_366, buf76, squeeze_175, buf74, primals_117, buf78, 512, 512, grid=grid(512, 512), stream=stream0)
        del buf73
        del convolution_58
        del primals_117
        del squeeze_175
        del unsqueeze_366
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf79 = aten.convolution_backward(buf78, where_57, primals_193, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_193
        buf80 = buf79[0]
        buf81 = buf79[1]
        del buf79
        buf82 = buf75; del buf75  # reuse
        buf84 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_11.run(where_57, buf80, convolution_57, unsqueeze_378, buf82, buf84, 2048, 128, grid=grid(2048), stream=stream0)
        buf83 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_6.run(buf82, buf83, 512, 4, grid=grid(512), stream=stream0)
        del buf82
        buf85 = empty((512, ), device='cuda', dtype=torch.float32)
        buf86 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_7.run(buf84, squeeze_172, buf85, buf86, 512, 4, grid=grid(512), stream=stream0)
        del buf84
        buf87 = buf78; del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_12.run(where_57, buf80, convolution_57, unsqueeze_378, buf85, squeeze_172, buf83, primals_115, buf87, 512, 512, grid=grid(512, 512), stream=stream0)
        del buf80
        del convolution_57
        del primals_115
        del squeeze_172
        del unsqueeze_378
        del where_57
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf88 = aten.convolution_backward(buf87, getitem_141, primals_192, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf87
        del getitem_141
        del primals_192
        buf89 = buf88[0]
        buf90 = buf88[1]
        del buf88
        buf91 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_22.run(buf91, buf19, buf37, buf54, buf71, buf89, 524288, grid=grid(524288), stream=stream0)
        del buf19
        del buf37
        del buf54
        del buf71
        del buf89
        buf92 = buf5; del buf5  # reuse
        buf94 = buf3; del buf3  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_23.run(gt_77, buf91, convolution_56, unsqueeze_390, buf92, buf94, 4096, 128, grid=grid(4096), stream=stream0)
        buf93 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_leaky_relu_backward_native_batch_norm_backward_2.run(buf92, buf93, 1024, 4, grid=grid(1024), stream=stream0)
        buf95 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf96 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_leaky_relu_backward_native_batch_norm_backward_3.run(buf94, squeeze_169, buf95, buf96, 1024, 4, grid=grid(1024), stream=stream0)
        buf97 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_24.run(gt_77, buf91, convolution_56, unsqueeze_390, buf95, squeeze_169, buf93, primals_113, buf97, 512, 1024, grid=grid(512, 1024), stream=stream0)
        del buf91
        del convolution_56
        del gt_77
        del primals_113
        del squeeze_169
        del unsqueeze_390
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf98 = aten.convolution_backward(buf97, where_55, primals_191, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_191
        buf99 = buf98[0]
        buf100 = buf98[1]
        del buf98
        buf101 = buf94; del buf94  # reuse
        buf103 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_25.run(where_55, buf99, convolution_55, unsqueeze_402, buf101, buf103, 4096, 128, grid=grid(4096), stream=stream0)
        buf102 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_leaky_relu_backward_native_batch_norm_backward_2.run(buf101, buf102, 1024, 4, grid=grid(1024), stream=stream0)
        del buf101
        buf104 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf105 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_leaky_relu_backward_native_batch_norm_backward_3.run(buf103, squeeze_166, buf104, buf105, 1024, 4, grid=grid(1024), stream=stream0)
        buf106 = buf97; del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_26.run(where_55, buf99, convolution_55, unsqueeze_402, buf104, squeeze_166, buf102, primals_111, buf106, 512, 1024, grid=grid(512, 1024), stream=stream0)
        del buf104
        del buf99
        del convolution_55
        del primals_111
        del squeeze_166
        del unsqueeze_402
        del where_55
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf107 = aten.convolution_backward(buf106, where_54, primals_190, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_190
        buf108 = buf107[0]
        buf109 = buf107[1]
        del buf107
        buf110 = reinterpret_tensor(buf0, (512, 16), (16, 1), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_27.run(where_54, buf108, buf110, 8192, 128, grid=grid(8192), stream=stream0)
        buf111 = buf85; del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_28.run(buf110, buf111, 512, 16, grid=grid(512), stream=stream0)
        buf112 = reinterpret_tensor(buf110, (512, 16), (1, 512), 0); del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_29.run(where_54, buf108, convolution_54, unsqueeze_414, buf112, 8192, 128, grid=grid(8192), stream=stream0)
        buf113 = empty((512, ), device='cuda', dtype=torch.float32)
        buf114 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_30.run(buf112, squeeze_163, buf113, buf114, 512, 16, grid=grid(512), stream=stream0)
        buf115 = empty_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_31.run(where_54, buf108, convolution_54, unsqueeze_414, buf113, squeeze_163, buf111, primals_109, buf115, 2048, 512, grid=grid(2048, 512), stream=stream0)
        del buf108
        del convolution_54
        del primals_109
        del squeeze_163
        del unsqueeze_414
        del where_54
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf116 = aten.convolution_backward(buf115, cat_3, primals_189, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_3
        del primals_189
        buf117 = buf116[0]
        buf118 = buf116[1]
        del buf116
        buf119 = reinterpret_tensor(buf103, (256, 16), (16, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_32.run(gt_80, buf117, buf119, 4096, 128, grid=grid(4096), stream=stream0)
        buf120 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_33.run(buf119, buf120, 256, 16, grid=grid(256), stream=stream0)
        buf121 = reinterpret_tensor(buf119, (256, 16), (1, 256), 0); del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_34.run(gt_80, buf117, convolution_53, unsqueeze_426, buf121, 4096, 128, grid=grid(4096), stream=stream0)
        buf122 = empty((256, ), device='cuda', dtype=torch.float32)
        buf123 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_35.run(buf121, squeeze_160, buf122, buf123, 256, 16, grid=grid(256), stream=stream0)
        buf124 = reinterpret_tensor(buf106, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_36.run(gt_80, buf117, convolution_53, unsqueeze_426, buf122, squeeze_160, buf120, primals_107, buf124, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del convolution_53
        del gt_80
        del primals_107
        del squeeze_160
        del unsqueeze_426
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf125 = aten.convolution_backward(buf124, add_283, primals_188, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_283
        del primals_188
        buf126 = buf125[0]
        buf127 = buf125[1]
        del buf125
        buf128 = reinterpret_tensor(buf121, (256, 16), (16, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_37.run(gt_81, buf126, buf128, 4096, 128, grid=grid(4096), stream=stream0)
        buf129 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_33.run(buf128, buf129, 256, 16, grid=grid(256), stream=stream0)
        buf130 = reinterpret_tensor(buf128, (256, 16), (1, 256), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_38.run(gt_81, buf126, convolution_52, unsqueeze_438, buf130, 4096, 128, grid=grid(4096), stream=stream0)
        buf131 = empty((256, ), device='cuda', dtype=torch.float32)
        buf132 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_35.run(buf130, squeeze_157, buf131, buf132, 256, 16, grid=grid(256), stream=stream0)
        buf133 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_39.run(gt_81, buf126, convolution_52, unsqueeze_438, buf131, squeeze_157, buf129, primals_105, buf133, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del convolution_52
        del gt_81
        del primals_105
        del squeeze_157
        del unsqueeze_438
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf134 = aten.convolution_backward(buf133, where_51, primals_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_187
        buf135 = buf134[0]
        buf136 = buf134[1]
        del buf134
        buf137 = reinterpret_tensor(buf130, (256, 16), (16, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_40.run(where_51, buf135, buf137, 4096, 128, grid=grid(4096), stream=stream0)
        buf138 = buf131; del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_33.run(buf137, buf138, 256, 16, grid=grid(256), stream=stream0)
        buf139 = reinterpret_tensor(buf137, (256, 16), (1, 256), 0); del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_41.run(where_51, buf135, convolution_51, unsqueeze_450, buf139, 4096, 128, grid=grid(4096), stream=stream0)
        buf140 = empty((256, ), device='cuda', dtype=torch.float32)
        buf141 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_35.run(buf139, squeeze_154, buf140, buf141, 256, 16, grid=grid(256), stream=stream0)
        buf142 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_42.run(where_51, buf135, convolution_51, unsqueeze_450, buf140, squeeze_154, buf138, primals_103, buf142, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del buf135
        del convolution_51
        del primals_103
        del squeeze_154
        del unsqueeze_450
        del where_51
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf143 = aten.convolution_backward(buf142, add_272, primals_186, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_272
        del primals_186
        buf144 = buf143[0]
        buf145 = buf143[1]
        del buf143
        buf146 = buf140; del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_43.run(gt_83, buf126, buf144, buf146, 256, 2048, grid=grid(256), stream=stream0)
        buf147 = reinterpret_tensor(buf139, (256, 16), (16, 1), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_44.run(gt_83, buf126, buf144, convolution_50, unsqueeze_462, buf147, 4096, 128, grid=grid(4096), stream=stream0)
        buf148 = empty((256, ), device='cuda', dtype=torch.float32)
        buf150 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_45.run(buf147, squeeze_151, buf148, buf150, 256, 16, grid=grid(256), stream=stream0)
        buf149 = buf142; del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_leaky_relu_backward_native_batch_norm_backward_46.run(gt_83, buf126, buf144, convolution_50, unsqueeze_462, buf148, squeeze_151, buf146, primals_101, buf149, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del convolution_50
        del gt_83
        del primals_101
        del squeeze_151
        del unsqueeze_462
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf151 = aten.convolution_backward(buf149, where_49, primals_185, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_185
        buf152 = buf151[0]
        buf153 = buf151[1]
        del buf151
        buf154 = buf147; del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_40.run(where_49, buf152, buf154, 4096, 128, grid=grid(4096), stream=stream0)
        buf155 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_33.run(buf154, buf155, 256, 16, grid=grid(256), stream=stream0)
        buf156 = reinterpret_tensor(buf154, (256, 16), (1, 256), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_41.run(where_49, buf152, convolution_49, unsqueeze_474, buf156, 4096, 128, grid=grid(4096), stream=stream0)
        buf157 = empty((256, ), device='cuda', dtype=torch.float32)
        buf158 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_35.run(buf156, squeeze_148, buf157, buf158, 256, 16, grid=grid(256), stream=stream0)
        buf159 = buf149; del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_42.run(where_49, buf152, convolution_49, unsqueeze_474, buf157, squeeze_148, buf155, primals_99, buf159, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del buf152
        del convolution_49
        del primals_99
        del squeeze_148
        del unsqueeze_474
        del where_49
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf160 = aten.convolution_backward(buf159, add_261, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_261
        del primals_184
        buf161 = buf160[0]
        buf162 = buf160[1]
        del buf160
        buf163 = buf157; del buf157  # reuse
        buf164 = empty((256, ), device='cuda', dtype=torch.float32)
        buf166 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_47.run(gt_85, buf126, buf144, buf161, convolution_48, unsqueeze_486, squeeze_145, buf163, buf164, buf166, 256, 2048, grid=grid(256), stream=stream0)
        buf167 = buf159; del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_48.run(gt_85, buf126, buf144, buf161, convolution_48, unsqueeze_486, buf164, squeeze_145, buf163, primals_97, buf167, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del convolution_48
        del gt_85
        del primals_97
        del squeeze_145
        del unsqueeze_486
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf168 = aten.convolution_backward(buf167, where_47, primals_183, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_183
        buf169 = buf168[0]
        buf170 = buf168[1]
        del buf168
        buf171 = reinterpret_tensor(buf156, (256, 16), (16, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_40.run(where_47, buf169, buf171, 4096, 128, grid=grid(4096), stream=stream0)
        buf172 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_33.run(buf171, buf172, 256, 16, grid=grid(256), stream=stream0)
        buf173 = reinterpret_tensor(buf171, (256, 16), (1, 256), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_41.run(where_47, buf169, convolution_47, unsqueeze_498, buf173, 4096, 128, grid=grid(4096), stream=stream0)
        buf174 = empty((256, ), device='cuda', dtype=torch.float32)
        buf175 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_35.run(buf173, squeeze_142, buf174, buf175, 256, 16, grid=grid(256), stream=stream0)
        buf176 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_42.run(where_47, buf169, convolution_47, unsqueeze_498, buf174, squeeze_142, buf172, primals_95, buf176, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del convolution_47
        del primals_95
        del squeeze_142
        del unsqueeze_498
        del where_47
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf177 = aten.convolution_backward(buf176, add_250, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_250
        del primals_182
        buf178 = buf177[0]
        buf179 = buf177[1]
        del buf177
        buf180 = reinterpret_tensor(buf176, (8, 256, 16, 16), (65536, 256, 16, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward]
        triton_poi_fused_add_leaky_relu_backward_49.run(gt_87, buf126, buf144, buf161, buf178, buf180, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del gt_87
        buf181 = buf174; del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_50.run(buf180, buf181, 256, 2048, grid=grid(256), stream=stream0)
        buf182 = reinterpret_tensor(buf173, (256, 16), (16, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_51.run(buf180, convolution_46, unsqueeze_510, buf182, 4096, 128, grid=grid(4096), stream=stream0)
        buf183 = empty((256, ), device='cuda', dtype=torch.float32)
        buf184 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_45.run(buf182, squeeze_139, buf183, buf184, 256, 16, grid=grid(256), stream=stream0)
        buf185 = reinterpret_tensor(buf169, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_52.run(buf180, convolution_46, unsqueeze_510, buf183, squeeze_139, buf181, primals_93, buf185, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del buf180
        del convolution_46
        del primals_93
        del squeeze_139
        del unsqueeze_510
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf186 = aten.convolution_backward(buf185, where_45, primals_181, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_181
        buf187 = buf186[0]
        buf188 = buf186[1]
        del buf186
        buf189 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_40.run(where_45, buf187, buf189, 4096, 128, grid=grid(4096), stream=stream0)
        buf190 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_33.run(buf189, buf190, 256, 16, grid=grid(256), stream=stream0)
        buf191 = reinterpret_tensor(buf189, (256, 16), (1, 256), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_41.run(where_45, buf187, convolution_45, unsqueeze_522, buf191, 4096, 128, grid=grid(4096), stream=stream0)
        buf192 = empty((256, ), device='cuda', dtype=torch.float32)
        buf193 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_35.run(buf191, squeeze_136, buf192, buf193, 256, 16, grid=grid(256), stream=stream0)
        buf194 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_42.run(where_45, buf187, convolution_45, unsqueeze_522, buf192, squeeze_136, buf190, primals_91, buf194, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del buf187
        del convolution_45
        del primals_91
        del squeeze_136
        del unsqueeze_522
        del where_45
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf195 = aten.convolution_backward(buf194, add_239, primals_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_239
        del buf194
        del primals_180
        buf196 = buf195[0]
        buf197 = buf195[1]
        del buf195
        buf198 = buf126; del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_53.run(buf198, buf144, buf161, buf178, buf196, 524288, grid=grid(524288), stream=stream0)
        del buf144
        del buf161
        del buf178
        buf199 = reinterpret_tensor(buf191, (256, 16), (16, 1), 0); del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_37.run(gt_89, buf198, buf199, 4096, 128, grid=grid(4096), stream=stream0)
        buf200 = buf192; del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_33.run(buf199, buf200, 256, 16, grid=grid(256), stream=stream0)
        buf201 = reinterpret_tensor(buf199, (256, 16), (1, 256), 0); del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_38.run(gt_89, buf198, convolution_44, unsqueeze_534, buf201, 4096, 128, grid=grid(4096), stream=stream0)
        buf202 = empty((256, ), device='cuda', dtype=torch.float32)
        buf203 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_35.run(buf201, squeeze_133, buf202, buf203, 256, 16, grid=grid(256), stream=stream0)
        buf204 = reinterpret_tensor(buf196, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_39.run(gt_89, buf198, convolution_44, unsqueeze_534, buf202, squeeze_133, buf200, primals_89, buf204, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del convolution_44
        del gt_89
        del primals_89
        del squeeze_133
        del unsqueeze_534
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf205 = aten.convolution_backward(buf204, where_43, primals_179, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_179
        buf206 = buf205[0]
        buf207 = buf205[1]
        del buf205
        buf208 = reinterpret_tensor(buf201, (256, 16), (16, 1), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_40.run(where_43, buf206, buf208, 4096, 128, grid=grid(4096), stream=stream0)
        buf209 = buf202; del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_33.run(buf208, buf209, 256, 16, grid=grid(256), stream=stream0)
        buf210 = reinterpret_tensor(buf208, (256, 16), (1, 256), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_41.run(where_43, buf206, convolution_43, unsqueeze_546, buf210, 4096, 128, grid=grid(4096), stream=stream0)
        buf211 = empty((256, ), device='cuda', dtype=torch.float32)
        buf212 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_35.run(buf210, squeeze_130, buf211, buf212, 256, 16, grid=grid(256), stream=stream0)
        buf213 = buf204; del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_42.run(where_43, buf206, convolution_43, unsqueeze_546, buf211, squeeze_130, buf209, primals_87, buf213, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del buf206
        del convolution_43
        del primals_87
        del squeeze_130
        del unsqueeze_546
        del where_43
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf214 = aten.convolution_backward(buf213, add_228, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_228
        del primals_178
        buf215 = buf214[0]
        buf216 = buf214[1]
        del buf214
        buf217 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_43.run(gt_91, buf198, buf215, buf217, 256, 2048, grid=grid(256), stream=stream0)
        buf218 = reinterpret_tensor(buf210, (256, 16), (16, 1), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_44.run(gt_91, buf198, buf215, convolution_42, unsqueeze_558, buf218, 4096, 128, grid=grid(4096), stream=stream0)
        buf219 = empty((256, ), device='cuda', dtype=torch.float32)
        buf221 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_45.run(buf218, squeeze_127, buf219, buf221, 256, 16, grid=grid(256), stream=stream0)
        buf220 = buf213; del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_leaky_relu_backward_native_batch_norm_backward_46.run(gt_91, buf198, buf215, convolution_42, unsqueeze_558, buf219, squeeze_127, buf217, primals_85, buf220, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del convolution_42
        del gt_91
        del primals_85
        del squeeze_127
        del unsqueeze_558
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf222 = aten.convolution_backward(buf220, where_41, primals_177, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_177
        buf223 = buf222[0]
        buf224 = buf222[1]
        del buf222
        buf225 = buf218; del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_40.run(where_41, buf223, buf225, 4096, 128, grid=grid(4096), stream=stream0)
        buf226 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_33.run(buf225, buf226, 256, 16, grid=grid(256), stream=stream0)
        buf227 = reinterpret_tensor(buf225, (256, 16), (1, 256), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_41.run(where_41, buf223, convolution_41, unsqueeze_570, buf227, 4096, 128, grid=grid(4096), stream=stream0)
        buf228 = empty((256, ), device='cuda', dtype=torch.float32)
        buf229 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_35.run(buf227, squeeze_124, buf228, buf229, 256, 16, grid=grid(256), stream=stream0)
        buf230 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_42.run(where_41, buf223, convolution_41, unsqueeze_570, buf228, squeeze_124, buf226, primals_83, buf230, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del buf223
        del convolution_41
        del primals_83
        del squeeze_124
        del unsqueeze_570
        del where_41
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf231 = aten.convolution_backward(buf230, add_217, primals_176, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_217
        del primals_176
        buf232 = buf231[0]
        buf233 = buf231[1]
        del buf231
        buf234 = buf228; del buf228  # reuse
        buf235 = empty((256, ), device='cuda', dtype=torch.float32)
        buf237 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_47.run(gt_93, buf198, buf215, buf232, convolution_40, unsqueeze_582, squeeze_121, buf234, buf235, buf237, 256, 2048, grid=grid(256), stream=stream0)
        buf238 = buf230; del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_48.run(gt_93, buf198, buf215, buf232, convolution_40, unsqueeze_582, buf235, squeeze_121, buf234, primals_81, buf238, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del convolution_40
        del gt_93
        del primals_81
        del squeeze_121
        del unsqueeze_582
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf239 = aten.convolution_backward(buf238, where_39, primals_175, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_175
        buf240 = buf239[0]
        buf241 = buf239[1]
        del buf239
        buf242 = reinterpret_tensor(buf227, (256, 16), (16, 1), 0); del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_40.run(where_39, buf240, buf242, 4096, 128, grid=grid(4096), stream=stream0)
        buf243 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_33.run(buf242, buf243, 256, 16, grid=grid(256), stream=stream0)
        buf244 = reinterpret_tensor(buf242, (256, 16), (1, 256), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_41.run(where_39, buf240, convolution_39, unsqueeze_594, buf244, 4096, 128, grid=grid(4096), stream=stream0)
        buf245 = empty((256, ), device='cuda', dtype=torch.float32)
        buf246 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_35.run(buf244, squeeze_118, buf245, buf246, 256, 16, grid=grid(256), stream=stream0)
        buf247 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_42.run(where_39, buf240, convolution_39, unsqueeze_594, buf245, squeeze_118, buf243, primals_79, buf247, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del convolution_39
        del primals_79
        del squeeze_118
        del unsqueeze_594
        del where_39
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf248 = aten.convolution_backward(buf247, add_206, primals_174, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_206
        del primals_174
        buf249 = buf248[0]
        buf250 = buf248[1]
        del buf248
        buf251 = reinterpret_tensor(buf247, (8, 256, 16, 16), (65536, 256, 16, 1), 0); del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward]
        triton_poi_fused_add_leaky_relu_backward_49.run(gt_95, buf198, buf215, buf232, buf249, buf251, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del gt_95
        buf252 = buf245; del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_50.run(buf251, buf252, 256, 2048, grid=grid(256), stream=stream0)
        buf253 = reinterpret_tensor(buf244, (256, 16), (16, 1), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_51.run(buf251, convolution_38, unsqueeze_606, buf253, 4096, 128, grid=grid(4096), stream=stream0)
        buf254 = empty((256, ), device='cuda', dtype=torch.float32)
        buf255 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_45.run(buf253, squeeze_115, buf254, buf255, 256, 16, grid=grid(256), stream=stream0)
        buf256 = reinterpret_tensor(buf240, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_52.run(buf251, convolution_38, unsqueeze_606, buf254, squeeze_115, buf252, primals_77, buf256, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del buf251
        del convolution_38
        del primals_77
        del squeeze_115
        del unsqueeze_606
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf257 = aten.convolution_backward(buf256, where_37, primals_173, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_173
        buf258 = buf257[0]
        buf259 = buf257[1]
        del buf257
        buf260 = buf253; del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_40.run(where_37, buf258, buf260, 4096, 128, grid=grid(4096), stream=stream0)
        buf261 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_33.run(buf260, buf261, 256, 16, grid=grid(256), stream=stream0)
        buf262 = reinterpret_tensor(buf260, (256, 16), (1, 256), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_41.run(where_37, buf258, convolution_37, unsqueeze_618, buf262, 4096, 128, grid=grid(4096), stream=stream0)
        buf263 = empty((256, ), device='cuda', dtype=torch.float32)
        buf264 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_35.run(buf262, squeeze_112, buf263, buf264, 256, 16, grid=grid(256), stream=stream0)
        del buf262
        buf265 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_42.run(where_37, buf258, convolution_37, unsqueeze_618, buf263, squeeze_112, buf261, primals_75, buf265, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del buf258
        del convolution_37
        del primals_75
        del squeeze_112
        del unsqueeze_618
        del where_37
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf266 = aten.convolution_backward(buf265, getitem_95, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf265
        del getitem_95
        del primals_172
        buf267 = buf266[0]
        buf268 = buf266[1]
        del buf266
        buf269 = buf117; del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_54.run(buf269, buf198, buf215, buf232, buf249, buf267, 1048576, grid=grid(1048576), stream=stream0)
        del buf198
        del buf215
        del buf232
        del buf249
        del buf267
        buf270 = reinterpret_tensor(buf112, (512, 16), (16, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_55.run(gt_97, buf269, buf270, 8192, 128, grid=grid(8192), stream=stream0)
        buf271 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_28.run(buf270, buf271, 512, 16, grid=grid(512), stream=stream0)
        buf272 = reinterpret_tensor(buf270, (512, 16), (1, 512), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_56.run(gt_97, buf269, convolution_36, unsqueeze_630, buf272, 8192, 128, grid=grid(8192), stream=stream0)
        buf273 = empty((512, ), device='cuda', dtype=torch.float32)
        buf274 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_30.run(buf272, squeeze_109, buf273, buf274, 512, 16, grid=grid(512), stream=stream0)
        buf275 = buf115; del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_57.run(gt_97, buf269, convolution_36, unsqueeze_630, buf273, squeeze_109, buf271, primals_73, buf275, 2048, 512, grid=grid(2048, 512), stream=stream0)
        del buf269
        del convolution_36
        del gt_97
        del primals_73
        del squeeze_109
        del unsqueeze_630
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf276 = aten.convolution_backward(buf275, where_35, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_171
        buf277 = buf276[0]
        buf278 = buf276[1]
        del buf276
        buf279 = reinterpret_tensor(buf272, (512, 16), (16, 1), 0); del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_27.run(where_35, buf277, buf279, 8192, 128, grid=grid(8192), stream=stream0)
        buf280 = buf273; del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_28.run(buf279, buf280, 512, 16, grid=grid(512), stream=stream0)
        buf281 = reinterpret_tensor(buf279, (512, 16), (1, 512), 0); del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_29.run(where_35, buf277, convolution_35, unsqueeze_642, buf281, 8192, 128, grid=grid(8192), stream=stream0)
        buf282 = empty((512, ), device='cuda', dtype=torch.float32)
        buf283 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_30.run(buf281, squeeze_106, buf282, buf283, 512, 16, grid=grid(512), stream=stream0)
        buf284 = buf275; del buf275  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_31.run(where_35, buf277, convolution_35, unsqueeze_642, buf282, squeeze_106, buf280, primals_71, buf284, 2048, 512, grid=grid(2048, 512), stream=stream0)
        del buf277
        del convolution_35
        del primals_71
        del squeeze_106
        del unsqueeze_642
        del where_35
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf285 = aten.convolution_backward(buf284, where_34, primals_170, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_170
        buf286 = buf285[0]
        buf287 = buf285[1]
        del buf285
        buf288 = empty((256, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_58.run(where_34, buf286, buf288, 16384, 128, grid=grid(16384), stream=stream0)
        buf289 = buf263; del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_59.run(buf288, buf289, 256, 64, grid=grid(256), stream=stream0)
        buf290 = reinterpret_tensor(buf288, (256, 64), (1, 256), 0); del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_60.run(where_34, buf286, convolution_34, unsqueeze_654, buf290, 16384, 128, grid=grid(16384), stream=stream0)
        buf291 = empty((256, ), device='cuda', dtype=torch.float32)
        buf292 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_61.run(buf290, squeeze_103, buf291, buf292, 256, 64, grid=grid(256), stream=stream0)
        buf293 = empty_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_62.run(where_34, buf286, convolution_34, unsqueeze_654, buf291, squeeze_103, buf289, primals_69, buf293, 8192, 256, grid=grid(8192, 256), stream=stream0)
        del buf286
        del convolution_34
        del primals_69
        del squeeze_103
        del unsqueeze_654
        del where_34
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf294 = aten.convolution_backward(buf293, cat_2, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_2
        del primals_169
        buf295 = buf294[0]
        buf296 = buf294[1]
        del buf294
        buf297 = reinterpret_tensor(buf281, (128, 64), (64, 1), 0); del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_63.run(gt_100, buf295, buf297, 8192, 128, grid=grid(8192), stream=stream0)
        buf298 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_64.run(buf297, buf298, 128, 64, grid=grid(128), stream=stream0)
        buf299 = reinterpret_tensor(buf297, (128, 64), (1, 128), 0); del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_65.run(gt_100, buf295, convolution_33, unsqueeze_666, buf299, 8192, 128, grid=grid(8192), stream=stream0)
        buf300 = empty((128, ), device='cuda', dtype=torch.float32)
        buf301 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_66.run(buf299, squeeze_100, buf300, buf301, 128, 64, grid=grid(128), stream=stream0)
        buf302 = reinterpret_tensor(buf284, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_67.run(gt_100, buf295, convolution_33, unsqueeze_666, buf300, squeeze_100, buf298, primals_67, buf302, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del convolution_33
        del gt_100
        del primals_67
        del squeeze_100
        del unsqueeze_666
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf303 = aten.convolution_backward(buf302, add_175, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_175
        del primals_168
        buf304 = buf303[0]
        buf305 = buf303[1]
        del buf303
        buf306 = reinterpret_tensor(buf299, (128, 64), (64, 1), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_68.run(gt_101, buf304, buf306, 8192, 128, grid=grid(8192), stream=stream0)
        buf307 = buf300; del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_64.run(buf306, buf307, 128, 64, grid=grid(128), stream=stream0)
        buf308 = reinterpret_tensor(buf306, (128, 64), (1, 128), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_69.run(gt_101, buf304, convolution_32, unsqueeze_678, buf308, 8192, 128, grid=grid(8192), stream=stream0)
        buf309 = empty((128, ), device='cuda', dtype=torch.float32)
        buf310 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_66.run(buf308, squeeze_97, buf309, buf310, 128, 64, grid=grid(128), stream=stream0)
        buf311 = buf302; del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_70.run(gt_101, buf304, convolution_32, unsqueeze_678, buf309, squeeze_97, buf307, primals_65, buf311, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del convolution_32
        del gt_101
        del primals_65
        del squeeze_97
        del unsqueeze_678
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf312 = aten.convolution_backward(buf311, where_31, primals_167, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_167
        buf313 = buf312[0]
        buf314 = buf312[1]
        del buf312
        buf315 = reinterpret_tensor(buf308, (128, 64), (64, 1), 0); del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_71.run(where_31, buf313, buf315, 8192, 128, grid=grid(8192), stream=stream0)
        buf316 = buf309; del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_64.run(buf315, buf316, 128, 64, grid=grid(128), stream=stream0)
        buf317 = reinterpret_tensor(buf315, (128, 64), (1, 128), 0); del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_72.run(where_31, buf313, convolution_31, unsqueeze_690, buf317, 8192, 128, grid=grid(8192), stream=stream0)
        buf318 = empty((128, ), device='cuda', dtype=torch.float32)
        buf319 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_66.run(buf317, squeeze_94, buf318, buf319, 128, 64, grid=grid(128), stream=stream0)
        buf320 = buf311; del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_73.run(where_31, buf313, convolution_31, unsqueeze_690, buf318, squeeze_94, buf316, primals_63, buf320, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del buf313
        del convolution_31
        del primals_63
        del squeeze_94
        del unsqueeze_690
        del where_31
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf321 = aten.convolution_backward(buf320, add_164, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_164
        del primals_166
        buf322 = buf321[0]
        buf323 = buf321[1]
        del buf321
        buf324 = buf318; del buf318  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_74.run(gt_103, buf304, buf322, buf324, 128, 8192, grid=grid(128), stream=stream0)
        buf325 = reinterpret_tensor(buf317, (128, 64), (64, 1), 0); del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_75.run(gt_103, buf304, buf322, convolution_30, unsqueeze_702, buf325, 8192, 128, grid=grid(8192), stream=stream0)
        buf326 = empty((128, ), device='cuda', dtype=torch.float32)
        buf328 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_76.run(buf325, squeeze_91, buf326, buf328, 128, 64, grid=grid(128), stream=stream0)
        buf327 = buf320; del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_leaky_relu_backward_native_batch_norm_backward_77.run(gt_103, buf304, buf322, convolution_30, unsqueeze_702, buf326, squeeze_91, buf324, primals_61, buf327, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del convolution_30
        del gt_103
        del primals_61
        del squeeze_91
        del unsqueeze_702
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf329 = aten.convolution_backward(buf327, where_29, primals_165, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_165
        buf330 = buf329[0]
        buf331 = buf329[1]
        del buf329
        buf332 = buf325; del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_71.run(where_29, buf330, buf332, 8192, 128, grid=grid(8192), stream=stream0)
        buf333 = buf326; del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_64.run(buf332, buf333, 128, 64, grid=grid(128), stream=stream0)
        buf334 = reinterpret_tensor(buf332, (128, 64), (1, 128), 0); del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_72.run(where_29, buf330, convolution_29, unsqueeze_714, buf334, 8192, 128, grid=grid(8192), stream=stream0)
        buf335 = empty((128, ), device='cuda', dtype=torch.float32)
        buf336 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_66.run(buf334, squeeze_88, buf335, buf336, 128, 64, grid=grid(128), stream=stream0)
        buf337 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_73.run(where_29, buf330, convolution_29, unsqueeze_714, buf335, squeeze_88, buf333, primals_59, buf337, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del buf330
        del convolution_29
        del primals_59
        del squeeze_88
        del unsqueeze_714
        del where_29
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf338 = aten.convolution_backward(buf337, add_153, primals_164, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_153
        del primals_164
        buf339 = buf338[0]
        buf340 = buf338[1]
        del buf338
        buf341 = buf335; del buf335  # reuse
        buf342 = empty((128, ), device='cuda', dtype=torch.float32)
        buf344 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_78.run(gt_105, buf304, buf322, buf339, convolution_28, unsqueeze_726, squeeze_85, buf341, buf342, buf344, 128, 8192, grid=grid(128), stream=stream0)
        buf345 = buf337; del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_79.run(gt_105, buf304, buf322, buf339, convolution_28, unsqueeze_726, buf342, squeeze_85, buf341, primals_57, buf345, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del convolution_28
        del gt_105
        del primals_57
        del squeeze_85
        del unsqueeze_726
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf346 = aten.convolution_backward(buf345, where_27, primals_163, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_163
        buf347 = buf346[0]
        buf348 = buf346[1]
        del buf346
        buf349 = reinterpret_tensor(buf334, (128, 64), (64, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_71.run(where_27, buf347, buf349, 8192, 128, grid=grid(8192), stream=stream0)
        buf350 = buf342; del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_64.run(buf349, buf350, 128, 64, grid=grid(128), stream=stream0)
        buf351 = reinterpret_tensor(buf349, (128, 64), (1, 128), 0); del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_72.run(where_27, buf347, convolution_27, unsqueeze_738, buf351, 8192, 128, grid=grid(8192), stream=stream0)
        buf352 = empty((128, ), device='cuda', dtype=torch.float32)
        buf353 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_66.run(buf351, squeeze_82, buf352, buf353, 128, 64, grid=grid(128), stream=stream0)
        buf354 = buf345; del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_73.run(where_27, buf347, convolution_27, unsqueeze_738, buf352, squeeze_82, buf350, primals_55, buf354, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del convolution_27
        del primals_55
        del squeeze_82
        del unsqueeze_738
        del where_27
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf355 = aten.convolution_backward(buf354, add_142, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_142
        del primals_162
        buf356 = buf355[0]
        buf357 = buf355[1]
        del buf355
        buf358 = reinterpret_tensor(buf354, (8, 128, 32, 32), (131072, 1024, 32, 1), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward]
        triton_poi_fused_add_leaky_relu_backward_80.run(gt_107, buf304, buf322, buf339, buf356, buf358, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del gt_107
        buf359 = buf352; del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf358, buf359, 128, 8192, grid=grid(128), stream=stream0)
        buf360 = reinterpret_tensor(buf351, (128, 64), (64, 1), 0); del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_82.run(buf358, convolution_26, unsqueeze_750, buf360, 8192, 128, grid=grid(8192), stream=stream0)
        buf361 = empty((128, ), device='cuda', dtype=torch.float32)
        buf362 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_76.run(buf360, squeeze_79, buf361, buf362, 128, 64, grid=grid(128), stream=stream0)
        buf363 = reinterpret_tensor(buf347, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_83.run(buf358, convolution_26, unsqueeze_750, buf361, squeeze_79, buf359, primals_53, buf363, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del buf358
        del convolution_26
        del primals_53
        del squeeze_79
        del unsqueeze_750
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf364 = aten.convolution_backward(buf363, where_25, primals_161, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_161
        buf365 = buf364[0]
        buf366 = buf364[1]
        del buf364
        buf367 = buf360; del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_71.run(where_25, buf365, buf367, 8192, 128, grid=grid(8192), stream=stream0)
        buf368 = buf361; del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_64.run(buf367, buf368, 128, 64, grid=grid(128), stream=stream0)
        buf369 = reinterpret_tensor(buf367, (128, 64), (1, 128), 0); del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_72.run(where_25, buf365, convolution_25, unsqueeze_762, buf369, 8192, 128, grid=grid(8192), stream=stream0)
        buf370 = empty((128, ), device='cuda', dtype=torch.float32)
        buf371 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_66.run(buf369, squeeze_76, buf370, buf371, 128, 64, grid=grid(128), stream=stream0)
        buf372 = buf363; del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_73.run(where_25, buf365, convolution_25, unsqueeze_762, buf370, squeeze_76, buf368, primals_51, buf372, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del buf365
        del convolution_25
        del primals_51
        del squeeze_76
        del unsqueeze_762
        del where_25
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf373 = aten.convolution_backward(buf372, add_131, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_131
        del buf372
        del primals_160
        buf374 = buf373[0]
        buf375 = buf373[1]
        del buf373
        buf376 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_84.run(buf376, buf322, buf339, buf356, buf374, 1048576, grid=grid(1048576), stream=stream0)
        del buf322
        del buf339
        del buf356
        buf377 = reinterpret_tensor(buf369, (128, 64), (64, 1), 0); del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_68.run(gt_109, buf376, buf377, 8192, 128, grid=grid(8192), stream=stream0)
        buf378 = buf370; del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_64.run(buf377, buf378, 128, 64, grid=grid(128), stream=stream0)
        buf379 = reinterpret_tensor(buf377, (128, 64), (1, 128), 0); del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_69.run(gt_109, buf376, convolution_24, unsqueeze_774, buf379, 8192, 128, grid=grid(8192), stream=stream0)
        buf380 = empty((128, ), device='cuda', dtype=torch.float32)
        buf381 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_66.run(buf379, squeeze_73, buf380, buf381, 128, 64, grid=grid(128), stream=stream0)
        buf382 = reinterpret_tensor(buf374, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_70.run(gt_109, buf376, convolution_24, unsqueeze_774, buf380, squeeze_73, buf378, primals_49, buf382, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del convolution_24
        del gt_109
        del primals_49
        del squeeze_73
        del unsqueeze_774
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf383 = aten.convolution_backward(buf382, where_23, primals_159, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_159
        buf384 = buf383[0]
        buf385 = buf383[1]
        del buf383
        buf386 = reinterpret_tensor(buf379, (128, 64), (64, 1), 0); del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_71.run(where_23, buf384, buf386, 8192, 128, grid=grid(8192), stream=stream0)
        buf387 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_64.run(buf386, buf387, 128, 64, grid=grid(128), stream=stream0)
        buf388 = reinterpret_tensor(buf386, (128, 64), (1, 128), 0); del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_72.run(where_23, buf384, convolution_23, unsqueeze_786, buf388, 8192, 128, grid=grid(8192), stream=stream0)
        buf389 = empty((128, ), device='cuda', dtype=torch.float32)
        buf390 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_66.run(buf388, squeeze_70, buf389, buf390, 128, 64, grid=grid(128), stream=stream0)
        buf391 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_73.run(where_23, buf384, convolution_23, unsqueeze_786, buf389, squeeze_70, buf387, primals_47, buf391, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del buf384
        del convolution_23
        del primals_47
        del squeeze_70
        del unsqueeze_786
        del where_23
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf392 = aten.convolution_backward(buf391, add_120, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_120
        del primals_158
        buf393 = buf392[0]
        buf394 = buf392[1]
        del buf392
        buf395 = buf389; del buf389  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_74.run(gt_111, buf376, buf393, buf395, 128, 8192, grid=grid(128), stream=stream0)
        buf396 = reinterpret_tensor(buf388, (128, 64), (64, 1), 0); del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_75.run(gt_111, buf376, buf393, convolution_22, unsqueeze_798, buf396, 8192, 128, grid=grid(8192), stream=stream0)
        buf397 = empty((128, ), device='cuda', dtype=torch.float32)
        buf399 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_76.run(buf396, squeeze_67, buf397, buf399, 128, 64, grid=grid(128), stream=stream0)
        buf398 = buf391; del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_leaky_relu_backward_native_batch_norm_backward_77.run(gt_111, buf376, buf393, convolution_22, unsqueeze_798, buf397, squeeze_67, buf395, primals_45, buf398, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del convolution_22
        del gt_111
        del primals_45
        del squeeze_67
        del unsqueeze_798
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf400 = aten.convolution_backward(buf398, where_21, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_157
        buf401 = buf400[0]
        buf402 = buf400[1]
        del buf400
        buf403 = buf396; del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_71.run(where_21, buf401, buf403, 8192, 128, grid=grid(8192), stream=stream0)
        buf404 = buf397; del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_64.run(buf403, buf404, 128, 64, grid=grid(128), stream=stream0)
        buf405 = reinterpret_tensor(buf403, (128, 64), (1, 128), 0); del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_72.run(where_21, buf401, convolution_21, unsqueeze_810, buf405, 8192, 128, grid=grid(8192), stream=stream0)
        buf406 = empty((128, ), device='cuda', dtype=torch.float32)
        buf407 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_66.run(buf405, squeeze_64, buf406, buf407, 128, 64, grid=grid(128), stream=stream0)
        buf408 = buf398; del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_73.run(where_21, buf401, convolution_21, unsqueeze_810, buf406, squeeze_64, buf404, primals_43, buf408, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del buf401
        del convolution_21
        del primals_43
        del squeeze_64
        del unsqueeze_810
        del where_21
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf409 = aten.convolution_backward(buf408, add_109, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_109
        del primals_156
        buf410 = buf409[0]
        buf411 = buf409[1]
        del buf409
        buf412 = buf406; del buf406  # reuse
        buf413 = empty((128, ), device='cuda', dtype=torch.float32)
        buf415 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_78.run(gt_113, buf376, buf393, buf410, convolution_20, unsqueeze_822, squeeze_61, buf412, buf413, buf415, 128, 8192, grid=grid(128), stream=stream0)
        buf416 = buf408; del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_79.run(gt_113, buf376, buf393, buf410, convolution_20, unsqueeze_822, buf413, squeeze_61, buf412, primals_41, buf416, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del convolution_20
        del gt_113
        del primals_41
        del squeeze_61
        del unsqueeze_822
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf417 = aten.convolution_backward(buf416, where_19, primals_155, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_155
        buf418 = buf417[0]
        buf419 = buf417[1]
        del buf417
        buf420 = reinterpret_tensor(buf405, (128, 64), (64, 1), 0); del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_71.run(where_19, buf418, buf420, 8192, 128, grid=grid(8192), stream=stream0)
        buf421 = buf413; del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_64.run(buf420, buf421, 128, 64, grid=grid(128), stream=stream0)
        buf422 = reinterpret_tensor(buf420, (128, 64), (1, 128), 0); del buf420  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_72.run(where_19, buf418, convolution_19, unsqueeze_834, buf422, 8192, 128, grid=grid(8192), stream=stream0)
        buf423 = empty((128, ), device='cuda', dtype=torch.float32)
        buf424 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_66.run(buf422, squeeze_58, buf423, buf424, 128, 64, grid=grid(128), stream=stream0)
        buf425 = buf416; del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_73.run(where_19, buf418, convolution_19, unsqueeze_834, buf423, squeeze_58, buf421, primals_39, buf425, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del convolution_19
        del primals_39
        del squeeze_58
        del unsqueeze_834
        del where_19
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf426 = aten.convolution_backward(buf425, add_98, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_98
        del primals_154
        buf427 = buf426[0]
        buf428 = buf426[1]
        del buf426
        buf429 = reinterpret_tensor(buf425, (8, 128, 32, 32), (131072, 1024, 32, 1), 0); del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward]
        triton_poi_fused_add_leaky_relu_backward_80.run(gt_115, buf376, buf393, buf410, buf427, buf429, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del gt_115
        buf430 = buf423; del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf429, buf430, 128, 8192, grid=grid(128), stream=stream0)
        buf431 = reinterpret_tensor(buf422, (128, 64), (64, 1), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_82.run(buf429, convolution_18, unsqueeze_846, buf431, 8192, 128, grid=grid(8192), stream=stream0)
        buf432 = empty((128, ), device='cuda', dtype=torch.float32)
        buf433 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_76.run(buf431, squeeze_55, buf432, buf433, 128, 64, grid=grid(128), stream=stream0)
        buf434 = reinterpret_tensor(buf418, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_83.run(buf429, convolution_18, unsqueeze_846, buf432, squeeze_55, buf430, primals_37, buf434, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del buf429
        del convolution_18
        del primals_37
        del squeeze_55
        del unsqueeze_846
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf435 = aten.convolution_backward(buf434, where_17, primals_153, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_153
        buf436 = buf435[0]
        buf437 = buf435[1]
        del buf435
        buf438 = buf431; del buf431  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_71.run(where_17, buf436, buf438, 8192, 128, grid=grid(8192), stream=stream0)
        buf439 = buf432; del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_64.run(buf438, buf439, 128, 64, grid=grid(128), stream=stream0)
        buf440 = reinterpret_tensor(buf438, (128, 64), (1, 128), 0); del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_72.run(where_17, buf436, convolution_17, unsqueeze_858, buf440, 8192, 128, grid=grid(8192), stream=stream0)
        buf441 = empty((128, ), device='cuda', dtype=torch.float32)
        buf442 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_66.run(buf440, squeeze_52, buf441, buf442, 128, 64, grid=grid(128), stream=stream0)
        del buf440
        buf443 = buf434; del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_73.run(where_17, buf436, convolution_17, unsqueeze_858, buf441, squeeze_52, buf439, primals_35, buf443, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del buf436
        del convolution_17
        del primals_35
        del squeeze_52
        del unsqueeze_858
        del where_17
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf444 = aten.convolution_backward(buf443, getitem_49, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf443
        del getitem_49
        del primals_152
        buf445 = buf444[0]
        buf446 = buf444[1]
        del buf444
        buf447 = buf295; del buf295  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_85.run(buf447, buf376, buf393, buf410, buf427, buf445, 2097152, grid=grid(2097152), stream=stream0)
        del buf376
        del buf393
        del buf410
        del buf427
        del buf445
        buf448 = reinterpret_tensor(buf290, (256, 64), (64, 1), 0); del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_86.run(gt_117, buf447, buf448, 16384, 128, grid=grid(16384), stream=stream0)
        buf449 = buf291; del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_59.run(buf448, buf449, 256, 64, grid=grid(256), stream=stream0)
        buf450 = reinterpret_tensor(buf448, (256, 64), (1, 256), 0); del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_87.run(gt_117, buf447, convolution_16, unsqueeze_870, buf450, 16384, 128, grid=grid(16384), stream=stream0)
        buf451 = empty((256, ), device='cuda', dtype=torch.float32)
        buf452 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_61.run(buf450, squeeze_49, buf451, buf452, 256, 64, grid=grid(256), stream=stream0)
        buf453 = buf293; del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_88.run(gt_117, buf447, convolution_16, unsqueeze_870, buf451, squeeze_49, buf449, primals_33, buf453, 8192, 256, grid=grid(8192, 256), stream=stream0)
        del buf447
        del convolution_16
        del gt_117
        del primals_33
        del squeeze_49
        del unsqueeze_870
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf454 = aten.convolution_backward(buf453, where_15, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_151
        buf455 = buf454[0]
        buf456 = buf454[1]
        del buf454
        buf457 = reinterpret_tensor(buf450, (256, 64), (64, 1), 0); del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_58.run(where_15, buf455, buf457, 16384, 128, grid=grid(16384), stream=stream0)
        buf458 = buf451; del buf451  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_59.run(buf457, buf458, 256, 64, grid=grid(256), stream=stream0)
        buf459 = reinterpret_tensor(buf457, (256, 64), (1, 256), 0); del buf457  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_60.run(where_15, buf455, convolution_15, unsqueeze_882, buf459, 16384, 128, grid=grid(16384), stream=stream0)
        buf460 = empty((256, ), device='cuda', dtype=torch.float32)
        buf461 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_61.run(buf459, squeeze_46, buf460, buf461, 256, 64, grid=grid(256), stream=stream0)
        buf462 = buf453; del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_62.run(where_15, buf455, convolution_15, unsqueeze_882, buf460, squeeze_46, buf458, primals_31, buf462, 8192, 256, grid=grid(8192, 256), stream=stream0)
        del buf455
        del convolution_15
        del primals_31
        del squeeze_46
        del unsqueeze_882
        del where_15
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf463 = aten.convolution_backward(buf462, where_14, primals_150, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_150
        buf464 = buf463[0]
        buf465 = buf463[1]
        del buf463
        buf466 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_89.run(where_14, buf464, buf466, 32768, 128, grid=grid(32768), stream=stream0)
        buf467 = buf441; del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_90.run(buf466, buf467, 128, 256, grid=grid(128), stream=stream0)
        buf468 = reinterpret_tensor(buf466, (128, 256), (1, 128), 0); del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_91.run(where_14, buf464, convolution_14, unsqueeze_894, buf468, 32768, 128, grid=grid(32768), stream=stream0)
        buf469 = empty((128, ), device='cuda', dtype=torch.float32)
        buf470 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_92.run(buf468, squeeze_43, buf469, buf470, 128, 256, grid=grid(128), stream=stream0)
        buf471 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_93.run(where_14, buf464, convolution_14, unsqueeze_894, buf469, squeeze_43, buf467, primals_29, buf471, 32768, 128, grid=grid(32768, 128), stream=stream0)
        del buf464
        del convolution_14
        del primals_29
        del squeeze_43
        del unsqueeze_894
        del where_14
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf472 = aten.convolution_backward(buf471, cat_1, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_1
        del primals_149
        buf473 = buf472[0]
        buf474 = buf472[1]
        del buf472
        buf475 = reinterpret_tensor(buf459, (64, 256), (256, 1), 0); del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_94.run(gt_120, buf473, buf475, 16384, 128, grid=grid(16384), stream=stream0)
        buf476 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_95.run(buf475, buf476, 64, 256, grid=grid(64), stream=stream0)
        buf477 = reinterpret_tensor(buf475, (64, 256), (1, 64), 0); del buf475  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_96.run(gt_120, buf473, convolution_13, unsqueeze_906, buf477, 16384, 128, grid=grid(16384), stream=stream0)
        buf478 = empty((64, ), device='cuda', dtype=torch.float32)
        buf479 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_97.run(buf477, squeeze_40, buf478, buf479, 64, 256, grid=grid(64), stream=stream0)
        buf480 = reinterpret_tensor(buf462, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_98.run(gt_120, buf473, convolution_13, unsqueeze_906, buf478, squeeze_40, buf476, primals_27, buf480, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del convolution_13
        del gt_120
        del primals_27
        del squeeze_40
        del unsqueeze_906
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf481 = aten.convolution_backward(buf480, add_67, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_67
        del primals_148
        buf482 = buf481[0]
        buf483 = buf481[1]
        del buf481
        buf484 = reinterpret_tensor(buf477, (64, 256), (256, 1), 0); del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_99.run(gt_121, buf482, buf484, 16384, 128, grid=grid(16384), stream=stream0)
        buf485 = buf478; del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_95.run(buf484, buf485, 64, 256, grid=grid(64), stream=stream0)
        buf486 = reinterpret_tensor(buf484, (64, 256), (1, 64), 0); del buf484  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_100.run(gt_121, buf482, convolution_12, unsqueeze_918, buf486, 16384, 128, grid=grid(16384), stream=stream0)
        buf487 = empty((64, ), device='cuda', dtype=torch.float32)
        buf488 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_97.run(buf486, squeeze_37, buf487, buf488, 64, 256, grid=grid(64), stream=stream0)
        buf489 = buf480; del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_101.run(gt_121, buf482, convolution_12, unsqueeze_918, buf487, squeeze_37, buf485, primals_25, buf489, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del convolution_12
        del gt_121
        del primals_25
        del squeeze_37
        del unsqueeze_918
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf490 = aten.convolution_backward(buf489, where_11, primals_147, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_147
        buf491 = buf490[0]
        buf492 = buf490[1]
        del buf490
        buf493 = reinterpret_tensor(buf486, (64, 256), (256, 1), 0); del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_102.run(where_11, buf491, buf493, 16384, 128, grid=grid(16384), stream=stream0)
        buf494 = buf487; del buf487  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_95.run(buf493, buf494, 64, 256, grid=grid(64), stream=stream0)
        buf495 = reinterpret_tensor(buf493, (64, 256), (1, 64), 0); del buf493  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_103.run(where_11, buf491, convolution_11, unsqueeze_930, buf495, 16384, 128, grid=grid(16384), stream=stream0)
        buf496 = empty((64, ), device='cuda', dtype=torch.float32)
        buf497 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_97.run(buf495, squeeze_34, buf496, buf497, 64, 256, grid=grid(64), stream=stream0)
        buf498 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_104.run(where_11, buf491, convolution_11, unsqueeze_930, buf496, squeeze_34, buf494, primals_23, buf498, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del buf491
        del convolution_11
        del primals_23
        del squeeze_34
        del unsqueeze_930
        del where_11
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf499 = aten.convolution_backward(buf498, add_56, primals_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_56
        del primals_146
        buf500 = buf499[0]
        buf501 = buf499[1]
        del buf499
        buf502 = reinterpret_tensor(buf460, (64, 4), (1, 64), 0); del buf460  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_105.run(gt_123, buf482, buf500, buf502, 256, 8192, grid=grid(256), stream=stream0)
        buf503 = buf496; del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_106.run(buf502, buf503, 64, 4, grid=grid(64), stream=stream0)
        del buf502
        buf504 = reinterpret_tensor(buf495, (64, 256), (256, 1), 0); del buf495  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_107.run(gt_123, buf482, buf500, convolution_10, unsqueeze_942, buf504, 16384, 128, grid=grid(16384), stream=stream0)
        buf505 = empty((64, ), device='cuda', dtype=torch.float32)
        buf507 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_leaky_relu_backward_native_batch_norm_backward_108.run(buf504, squeeze_31, buf505, buf507, 64, 256, grid=grid(64), stream=stream0)
        buf506 = buf498; del buf498  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_leaky_relu_backward_native_batch_norm_backward_109.run(gt_123, buf482, buf500, convolution_10, unsqueeze_942, buf505, squeeze_31, buf503, primals_21, buf506, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del convolution_10
        del gt_123
        del primals_21
        del squeeze_31
        del unsqueeze_942
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf508 = aten.convolution_backward(buf506, where_9, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_145
        buf509 = buf508[0]
        buf510 = buf508[1]
        del buf508
        buf511 = buf504; del buf504  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_102.run(where_9, buf509, buf511, 16384, 128, grid=grid(16384), stream=stream0)
        buf512 = buf505; del buf505  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_95.run(buf511, buf512, 64, 256, grid=grid(64), stream=stream0)
        buf513 = reinterpret_tensor(buf511, (64, 256), (1, 64), 0); del buf511  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_103.run(where_9, buf509, convolution_9, unsqueeze_954, buf513, 16384, 128, grid=grid(16384), stream=stream0)
        buf514 = empty((64, ), device='cuda', dtype=torch.float32)
        buf515 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_97.run(buf513, squeeze_28, buf514, buf515, 64, 256, grid=grid(64), stream=stream0)
        del buf513
        buf516 = buf506; del buf506  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_104.run(where_9, buf509, convolution_9, unsqueeze_954, buf514, squeeze_28, buf512, primals_19, buf516, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del buf509
        del convolution_9
        del primals_19
        del squeeze_28
        del unsqueeze_954
        del where_9
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf517 = aten.convolution_backward(buf516, getitem_27, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf516
        del getitem_27
        del primals_144
        buf518 = buf517[0]
        buf519 = buf517[1]
        del buf517
        buf520 = buf473; del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.leaky_relu_backward]
        triton_poi_fused_cat_leaky_relu_backward_110.run(buf520, gt_125, buf482, buf500, buf518, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        del buf482
        del buf500
        del buf518
        del gt_125
        buf521 = reinterpret_tensor(buf282, (128, 4), (1, 128), 0); del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_111.run(buf520, buf521, 512, 8192, grid=grid(512), stream=stream0)
        buf522 = buf469; del buf469  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_112.run(buf521, buf522, 128, 4, grid=grid(128), stream=stream0)
        buf523 = reinterpret_tensor(buf468, (128, 256), (256, 1), 0); del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_113.run(buf520, convolution_8, unsqueeze_966, buf523, 32768, 128, grid=grid(32768), stream=stream0)
        buf524 = empty((128, ), device='cuda', dtype=torch.float32)
        buf525 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_114.run(buf523, squeeze_25, buf524, buf525, 128, 256, grid=grid(128), stream=stream0)
        buf526 = buf471; del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_115.run(buf520, convolution_8, unsqueeze_966, buf524, squeeze_25, buf522, primals_17, buf526, 32768, 128, grid=grid(32768, 128), stream=stream0)
        del buf520
        del convolution_8
        del primals_17
        del squeeze_25
        del unsqueeze_966
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf527 = aten.convolution_backward(buf526, where_7, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_143
        buf528 = buf527[0]
        buf529 = buf527[1]
        del buf527
        buf530 = buf523; del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_89.run(where_7, buf528, buf530, 32768, 128, grid=grid(32768), stream=stream0)
        buf531 = buf524; del buf524  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_90.run(buf530, buf531, 128, 256, grid=grid(128), stream=stream0)
        buf532 = reinterpret_tensor(buf530, (128, 256), (1, 128), 0); del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_91.run(where_7, buf528, convolution_7, unsqueeze_978, buf532, 32768, 128, grid=grid(32768), stream=stream0)
        buf533 = empty((128, ), device='cuda', dtype=torch.float32)
        buf534 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_92.run(buf532, squeeze_22, buf533, buf534, 128, 256, grid=grid(128), stream=stream0)
        buf535 = buf526; del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_93.run(where_7, buf528, convolution_7, unsqueeze_978, buf533, squeeze_22, buf531, primals_15, buf535, 32768, 128, grid=grid(32768, 128), stream=stream0)
        del buf528
        del convolution_7
        del primals_15
        del squeeze_22
        del unsqueeze_978
        del where_7
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf536 = aten.convolution_backward(buf535, where_6, primals_142, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_142
        buf537 = buf536[0]
        buf538 = buf536[1]
        del buf536
        buf539 = empty((64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_116.run(where_6, buf537, buf539, 65536, 128, grid=grid(65536), stream=stream0)
        buf540 = buf514; del buf514  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_117.run(buf539, buf540, 64, 1024, grid=grid(64), stream=stream0)
        buf541 = reinterpret_tensor(buf539, (64, 1024), (1, 64), 0); del buf539  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_118.run(where_6, buf537, convolution_6, unsqueeze_990, buf541, 65536, 128, grid=grid(65536), stream=stream0)
        buf542 = empty((64, ), device='cuda', dtype=torch.float32)
        buf543 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_119.run(buf541, squeeze_19, buf542, buf543, 64, 1024, grid=grid(64), stream=stream0)
        buf544 = empty_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_120.run(where_6, buf537, convolution_6, unsqueeze_990, buf542, squeeze_19, buf540, primals_13, buf544, 131072, 64, grid=grid(131072, 64), stream=stream0)
        del buf537
        del convolution_6
        del primals_13
        del squeeze_19
        del unsqueeze_990
        del where_6
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf545 = aten.convolution_backward(buf544, cat, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat
        del primals_141
        buf546 = buf545[0]
        buf547 = buf545[1]
        del buf545
        buf548 = reinterpret_tensor(buf541, (64, 1024), (1024, 1), 0); del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_121.run(gt_128, buf546, buf548, 65536, 128, grid=grid(65536), stream=stream0)
        buf549 = buf542; del buf542  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_117.run(buf548, buf549, 64, 1024, grid=grid(64), stream=stream0)
        buf550 = reinterpret_tensor(buf548, (64, 1024), (1, 64), 0); del buf548  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_122.run(gt_128, buf546, convolution_5, unsqueeze_1002, buf550, 65536, 128, grid=grid(65536), stream=stream0)
        buf551 = empty((64, ), device='cuda', dtype=torch.float32)
        buf552 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_119.run(buf550, squeeze_16, buf551, buf552, 64, 1024, grid=grid(64), stream=stream0)
        buf553 = buf544; del buf544  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_123.run(gt_128, buf546, convolution_5, unsqueeze_1002, buf551, squeeze_16, buf549, primals_11, buf553, 131072, 64, grid=grid(131072, 64), stream=stream0)
        del convolution_5
        del gt_128
        del primals_11
        del squeeze_16
        del unsqueeze_1002
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf554 = aten.convolution_backward(buf553, add_25, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_25
        del primals_140
        buf555 = buf554[0]
        buf556 = buf554[1]
        del buf554
        buf557 = reinterpret_tensor(buf550, (64, 1024), (1024, 1), 0); del buf550  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_124.run(gt_129, buf555, buf557, 65536, 128, grid=grid(65536), stream=stream0)
        buf558 = buf551; del buf551  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_117.run(buf557, buf558, 64, 1024, grid=grid(64), stream=stream0)
        buf559 = reinterpret_tensor(buf557, (64, 1024), (1, 64), 0); del buf557  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_125.run(gt_129, buf555, convolution_4, unsqueeze_1014, buf559, 65536, 128, grid=grid(65536), stream=stream0)
        buf560 = empty((64, ), device='cuda', dtype=torch.float32)
        buf561 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_119.run(buf559, squeeze_13, buf560, buf561, 64, 1024, grid=grid(64), stream=stream0)
        buf562 = buf553; del buf553  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_126.run(gt_129, buf555, convolution_4, unsqueeze_1014, buf560, squeeze_13, buf558, primals_9, buf562, 131072, 64, grid=grid(131072, 64), stream=stream0)
        del convolution_4
        del gt_129
        del primals_9
        del squeeze_13
        del unsqueeze_1014
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf563 = aten.convolution_backward(buf562, where_3, primals_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf562
        del primals_139
        buf564 = buf563[0]
        buf565 = buf563[1]
        del buf563
        buf566 = reinterpret_tensor(buf532, (32, 1024), (1024, 1), 0); del buf532  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_127.run(where_3, buf564, buf566, 32768, 128, grid=grid(32768), stream=stream0)
        buf567 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_128.run(buf566, buf567, 32, 1024, grid=grid(32), stream=stream0)
        buf568 = reinterpret_tensor(buf566, (32, 1024), (1, 32), 0); del buf566  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_129.run(where_3, buf564, convolution_3, unsqueeze_1026, buf568, 32768, 128, grid=grid(32768), stream=stream0)
        buf569 = empty((32, ), device='cuda', dtype=torch.float32)
        buf570 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_130.run(buf568, squeeze_10, buf569, buf570, 32, 1024, grid=grid(32), stream=stream0)
        buf571 = reinterpret_tensor(buf535, (8, 32, 128, 128), (524288, 1, 4096, 32), 0); del buf535  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_131.run(where_3, buf564, convolution_3, unsqueeze_1026, buf569, squeeze_10, buf567, primals_7, buf571, 131072, 32, grid=grid(131072, 32), stream=stream0)
        del buf564
        del convolution_3
        del primals_7
        del squeeze_10
        del unsqueeze_1026
        del where_3
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf572 = aten.convolution_backward(buf571, getitem_9, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf571
        del getitem_9
        del primals_138
        buf573 = buf572[0]
        buf574 = buf572[1]
        del buf572
        buf575 = buf521; del buf521  # reuse
        buf577 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_cat_leaky_relu_backward_native_batch_norm_backward_132.run(gt_131, buf546, buf555, buf573, convolution_2, unsqueeze_1038, buf575, buf577, 512, 32768, grid=grid(512), stream=stream0)
        buf576 = buf533; del buf533  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_112.run(buf575, buf576, 128, 4, grid=grid(128), stream=stream0)
        del buf575
        buf578 = empty((128, ), device='cuda', dtype=torch.float32)
        buf580 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_cat_leaky_relu_backward_native_batch_norm_backward_133.run(buf577, squeeze_7, buf578, buf580, 128, 4, grid=grid(128), stream=stream0)
        del buf577
        buf579 = buf546; del buf546  # reuse
        buf581 = empty_strided((8, 128, 128, 128), (2097152, 1, 16384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_cat_convolution_backward_leaky_relu_backward_native_batch_norm_backward_134.run(buf579, gt_131, buf555, buf573, convolution_2, unsqueeze_1038, buf578, squeeze_7, buf576, primals_5, buf581, 1024, 16384, grid=grid(1024, 16384), stream=stream0)
        del buf555
        del buf578
        del buf579
        del convolution_2
        del gt_131
        del primals_5
        del squeeze_7
        del unsqueeze_1038
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf582 = aten.convolution_backward(buf581, where_1, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_137
        buf583 = buf582[0]
        buf584 = buf582[1]
        del buf582
        buf585 = reinterpret_tensor(buf559, (64, 1024), (1024, 1), 0); del buf559  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_116.run(where_1, buf583, buf585, 65536, 128, grid=grid(65536), stream=stream0)
        buf586 = buf560; del buf560  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_117.run(buf585, buf586, 64, 1024, grid=grid(64), stream=stream0)
        buf587 = reinterpret_tensor(buf585, (64, 1024), (1, 64), 0); del buf585  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_118.run(where_1, buf583, convolution_1, unsqueeze_1050, buf587, 65536, 128, grid=grid(65536), stream=stream0)
        buf588 = empty((64, ), device='cuda', dtype=torch.float32)
        buf589 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_119.run(buf587, squeeze_4, buf588, buf589, 64, 1024, grid=grid(64), stream=stream0)
        del buf587
        buf590 = reinterpret_tensor(buf573, (8, 64, 128, 128), (1048576, 1, 8192, 64), 0); del buf573  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_120.run(where_1, buf583, convolution_1, unsqueeze_1050, buf588, squeeze_4, buf586, primals_3, buf590, 131072, 64, grid=grid(131072, 64), stream=stream0)
        del buf583
        del buf588
        del convolution_1
        del primals_3
        del squeeze_4
        del unsqueeze_1050
        del where_1
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf591 = aten.convolution_backward(buf590, where, primals_136, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf590
        del primals_136
        buf592 = buf591[0]
        buf593 = buf591[1]
        del buf591
        buf594 = reinterpret_tensor(buf568, (32, 1024), (1024, 1), 0); del buf568  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_135.run(where, buf592, buf594, 32768, 512, grid=grid(32768), stream=stream0)
        buf595 = buf569; del buf569  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_per_fused_leaky_relu_backward_native_batch_norm_backward_128.run(buf594, buf595, 32, 1024, grid=grid(32), stream=stream0)
        buf596 = reinterpret_tensor(buf594, (32, 1024), (1, 32), 0); del buf594  # reuse
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_136.run(where, buf592, convolution, unsqueeze_1062, buf596, 32768, 512, grid=grid(32768), stream=stream0)
        buf597 = empty((32, ), device='cuda', dtype=torch.float32)
        buf598 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_130.run(buf596, squeeze_1, buf597, buf598, 32, 1024, grid=grid(32), stream=stream0)
        del buf596
        buf599 = reinterpret_tensor(buf581, (8, 32, 256, 256), (2097152, 1, 8192, 32), 0); del buf581  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_137.run(where, buf592, convolution, unsqueeze_1062, buf597, squeeze_1, buf595, primals_1, buf599, 524288, 32, grid=grid(524288, 32), stream=stream0)
        del buf592
        del buf597
        del convolution
        del primals_1
        del squeeze_1
        del unsqueeze_1062
        del where
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
        buf600 = aten.convolution_backward(buf599, primals_405, primals_135, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf599
        del primals_135
        del primals_405
        buf601 = buf600[1]
        return (buf598, buf595, buf589, buf586, buf580, buf576, buf570, buf567, buf561, buf558, buf552, buf549, buf543, buf540, buf534, buf531, buf525, buf522, buf515, buf512, buf507, buf503, buf497, buf494, buf488, buf485, buf479, buf476, buf470, buf467, buf461, buf458, buf452, buf449, buf442, buf439, buf433, buf430, buf424, buf421, buf415, buf412, buf407, buf404, buf399, buf395, buf390, buf387, buf381, buf378, buf371, buf368, buf362, buf359, buf353, buf350, buf344, buf341, buf336, buf333, buf328, buf324, buf319, buf316, buf310, buf307, buf301, buf298, buf292, buf289, buf283, buf280, buf274, buf271, buf264, buf261, buf255, buf252, buf246, buf243, buf237, buf234, buf229, buf226, buf221, buf217, buf212, buf209, buf203, buf200, buf193, buf190, buf184, buf181, buf175, buf172, buf166, buf163, buf158, buf155, buf150, buf146, buf141, buf138, buf132, buf129, buf123, buf120, buf114, buf111, buf105, buf102, buf96, buf93, buf86, buf83, buf77, buf74, buf68, buf65, buf59, buf56, buf51, buf48, buf43, buf39, buf34, buf31, buf25, buf22, buf16, buf13, buf7, buf4, buf601, buf593, buf584, buf574, buf565, buf556, buf547, buf538, buf529, buf519, buf510, buf501, buf492, buf483, buf474, buf465, buf456, buf446, buf437, buf428, buf419, buf411, buf402, buf394, buf385, buf375, buf366, buf357, buf348, buf340, buf331, buf323, buf314, buf305, buf296, buf287, buf278, buf268, buf259, buf250, buf241, buf233, buf224, buf216, buf207, buf197, buf188, buf179, buf170, buf162, buf153, buf145, buf136, buf127, buf118, buf109, buf100, buf90, buf81, buf72, buf63, buf55, buf46, buf38, buf29, buf20, buf11, reinterpret_tensor(buf1, (1000, 1024), (1024, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1024, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 32, 256, 256), (2097152, 1, 8192, 32), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    where = rand_strided((8, 32, 256, 256), (2097152, 1, 8192, 32), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_1 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 128, 128, 128), (2097152, 1, 16384, 128), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_9 = rand_strided((8, 64, 128, 128), (2097152, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_3 = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_25 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((8, 128, 128, 128), (2097152, 1, 16384, 128), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_6 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_7 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_27 = rand_strided((8, 64, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_9 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_56 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_11 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_67 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_14 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_15 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_49 = rand_strided((8, 128, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_17 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_98 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_19 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_109 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_21 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_120 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_23 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_131 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_25 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_142 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_27 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_153 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_29 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_164 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_31 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_175 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_34 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_35 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_95 = rand_strided((8, 256, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_37 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_206 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_39 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_217 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_41 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_228 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_43 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_239 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_45 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_250 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_47 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_261 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_49 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_272 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_51 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_157 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_283 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_160 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_3 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda:0', dtype=torch.float32)
    squeeze_163 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_54 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_166 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_55 = rand_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_169 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_141 = rand_strided((8, 512, 8, 8), (65536, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_57 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    squeeze_172 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_57 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    convolution_58 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    squeeze_175 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_314 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    convolution_59 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    squeeze_178 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_59 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    squeeze_181 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_325 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    squeeze_184 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_61 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    convolution_62 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    squeeze_187 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_336 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    convolution_63 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    squeeze_190 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    where_63 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    convolution_64 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    squeeze_193 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_347 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    convolution_65 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    squeeze_196 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_4 = rand_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cuda:0', dtype=torch.float32)
    convolution_66 = rand_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_199 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((8, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    gt_67 = rand_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cuda:0', dtype=torch.bool)
    unsqueeze_270 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_68 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.bool)
    unsqueeze_282 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_69 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.bool)
    unsqueeze_294 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_306 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_71 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.bool)
    unsqueeze_318 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_73 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.bool)
    unsqueeze_342 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_354 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_75 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.bool)
    unsqueeze_366 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_378 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_77 = rand_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cuda:0', dtype=torch.bool)
    unsqueeze_390 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_414 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_80 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bool)
    unsqueeze_426 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_81 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bool)
    unsqueeze_438 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_450 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_83 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bool)
    unsqueeze_462 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_85 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bool)
    unsqueeze_486 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_498 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_87 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bool)
    unsqueeze_510 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_522 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_89 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bool)
    unsqueeze_534 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_546 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_91 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bool)
    unsqueeze_558 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_570 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_93 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bool)
    unsqueeze_582 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_594 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_95 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bool)
    unsqueeze_606 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_618 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_97 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda:0', dtype=torch.bool)
    unsqueeze_630 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_642 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_654 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_100 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bool)
    unsqueeze_666 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_101 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bool)
    unsqueeze_678 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_690 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_103 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bool)
    unsqueeze_702 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_714 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_105 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bool)
    unsqueeze_726 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_738 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_107 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bool)
    unsqueeze_750 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_762 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_109 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bool)
    unsqueeze_774 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_786 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_111 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bool)
    unsqueeze_798 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_810 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_113 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bool)
    unsqueeze_822 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_834 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_115 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bool)
    unsqueeze_846 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_858 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_117 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cuda:0', dtype=torch.bool)
    unsqueeze_870 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_882 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_894 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_120 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_906 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_121 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_918 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_930 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_123 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_942 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_954 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_125 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.bool)
    unsqueeze_966 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_978 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_990 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_128 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_1002 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_129 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_1014 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1026 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_131 = rand_strided((8, 128, 128, 128), (2097152, 1, 16384, 128), device='cuda:0', dtype=torch.bool)
    unsqueeze_1038 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1050 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1062 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_405, convolution, squeeze_1, where, convolution_1, squeeze_4, where_1, convolution_2, squeeze_7, getitem_9, convolution_3, squeeze_10, where_3, convolution_4, squeeze_13, add_25, convolution_5, squeeze_16, cat, convolution_6, squeeze_19, where_6, convolution_7, squeeze_22, where_7, convolution_8, squeeze_25, getitem_27, convolution_9, squeeze_28, where_9, convolution_10, squeeze_31, add_56, convolution_11, squeeze_34, where_11, convolution_12, squeeze_37, add_67, convolution_13, squeeze_40, cat_1, convolution_14, squeeze_43, where_14, convolution_15, squeeze_46, where_15, convolution_16, squeeze_49, getitem_49, convolution_17, squeeze_52, where_17, convolution_18, squeeze_55, add_98, convolution_19, squeeze_58, where_19, convolution_20, squeeze_61, add_109, convolution_21, squeeze_64, where_21, convolution_22, squeeze_67, add_120, convolution_23, squeeze_70, where_23, convolution_24, squeeze_73, add_131, convolution_25, squeeze_76, where_25, convolution_26, squeeze_79, add_142, convolution_27, squeeze_82, where_27, convolution_28, squeeze_85, add_153, convolution_29, squeeze_88, where_29, convolution_30, squeeze_91, add_164, convolution_31, squeeze_94, where_31, convolution_32, squeeze_97, add_175, convolution_33, squeeze_100, cat_2, convolution_34, squeeze_103, where_34, convolution_35, squeeze_106, where_35, convolution_36, squeeze_109, getitem_95, convolution_37, squeeze_112, where_37, convolution_38, squeeze_115, add_206, convolution_39, squeeze_118, where_39, convolution_40, squeeze_121, add_217, convolution_41, squeeze_124, where_41, convolution_42, squeeze_127, add_228, convolution_43, squeeze_130, where_43, convolution_44, squeeze_133, add_239, convolution_45, squeeze_136, where_45, convolution_46, squeeze_139, add_250, convolution_47, squeeze_142, where_47, convolution_48, squeeze_145, add_261, convolution_49, squeeze_148, where_49, convolution_50, squeeze_151, add_272, convolution_51, squeeze_154, where_51, convolution_52, squeeze_157, add_283, convolution_53, squeeze_160, cat_3, convolution_54, squeeze_163, where_54, convolution_55, squeeze_166, where_55, convolution_56, squeeze_169, getitem_141, convolution_57, squeeze_172, where_57, convolution_58, squeeze_175, add_314, convolution_59, squeeze_178, where_59, convolution_60, squeeze_181, add_325, convolution_61, squeeze_184, where_61, convolution_62, squeeze_187, add_336, convolution_63, squeeze_190, where_63, convolution_64, squeeze_193, add_347, convolution_65, squeeze_196, cat_4, convolution_66, squeeze_199, clone, permute_1, gt_67, unsqueeze_270, gt_68, unsqueeze_282, gt_69, unsqueeze_294, unsqueeze_306, gt_71, unsqueeze_318, unsqueeze_330, gt_73, unsqueeze_342, unsqueeze_354, gt_75, unsqueeze_366, unsqueeze_378, gt_77, unsqueeze_390, unsqueeze_402, unsqueeze_414, gt_80, unsqueeze_426, gt_81, unsqueeze_438, unsqueeze_450, gt_83, unsqueeze_462, unsqueeze_474, gt_85, unsqueeze_486, unsqueeze_498, gt_87, unsqueeze_510, unsqueeze_522, gt_89, unsqueeze_534, unsqueeze_546, gt_91, unsqueeze_558, unsqueeze_570, gt_93, unsqueeze_582, unsqueeze_594, gt_95, unsqueeze_606, unsqueeze_618, gt_97, unsqueeze_630, unsqueeze_642, unsqueeze_654, gt_100, unsqueeze_666, gt_101, unsqueeze_678, unsqueeze_690, gt_103, unsqueeze_702, unsqueeze_714, gt_105, unsqueeze_726, unsqueeze_738, gt_107, unsqueeze_750, unsqueeze_762, gt_109, unsqueeze_774, unsqueeze_786, gt_111, unsqueeze_798, unsqueeze_810, gt_113, unsqueeze_822, unsqueeze_834, gt_115, unsqueeze_846, unsqueeze_858, gt_117, unsqueeze_870, unsqueeze_882, unsqueeze_894, gt_120, unsqueeze_906, gt_121, unsqueeze_918, unsqueeze_930, gt_123, unsqueeze_942, unsqueeze_954, gt_125, unsqueeze_966, unsqueeze_978, unsqueeze_990, gt_128, unsqueeze_1002, gt_129, unsqueeze_1014, unsqueeze_1026, gt_131, unsqueeze_1038, unsqueeze_1050, unsqueeze_1062, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('cspdarknet53', benchmark_compiled_module)
