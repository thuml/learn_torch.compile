
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


# kernel path: /tmp/torchinductor_youkaichao/xv/cxvcnbuskdlhxg5prrlfbs7rhi5ldt35fj4pfpmnu4yvjsdtbyyn.py
# Source Nodes: [div__45], Original ATen: [aten.div, aten.mul, aten.native_layer_norm_backward]
# div__45 => div_69
triton_red_fused_div_mul_native_layer_norm_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_mul_native_layer_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 49)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1024*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r2 + (1024*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 49.0
        tmp2 = tmp0 / tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp9 = tmp4 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp13 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp14 = tl.load(in_ptr0 + (r2 + (1024*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr2 + (r2 + (1024*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = 49.0
        tmp16 = tmp14 / tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = 1024.0
        tmp20 = tmp18 * tmp19
        tmp21 = tmp20 - tmp6
        tmp23 = tmp22 * tmp11
        tmp24 = tmp21 - tmp23
        tmp25 = tmp13 * tmp24
        tmp27 = 0.8999999985098839
        tmp28 = tmp26 / tmp27
        tmp29 = tmp25 * tmp28
        tl.store(out_ptr2 + (r2 + (1024*x3)), tmp25, rmask & xmask)
        tl.store(out_ptr3 + (r2 + (1024*x3)), tmp29, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ag/cagv3oik7dq47gl3jdent4zwek3s7mtqivo6yel42ijixtqnz7dz.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]

triton_red_fused_div_native_layer_norm_backward_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_layer_norm_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*(r2 // 49)) + (2048*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 49.0
        tmp2 = tmp0 / tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h4/ch4pdueuusjx2en2hosw3kbqfkbw2mxh6nklbzf22b54hqqgad7m.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]

triton_per_fused_div_native_layer_norm_backward_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_layer_norm_backward_3', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/pz/cpzdsubduqevhudvk7n3yrymqvjc5nlkfenbpym24jl5sfoq2uzp.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]

triton_red_fused_div_native_layer_norm_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_layer_norm_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 49.0
        tmp2 = tmp0 / tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iu/ciu6d2muh7z77tqnfswqkmo4lekr4wjrxcpgvh5r2gwri4jsio7j.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lp/clpyzv77rloe7tjcnaswujx6hgemccateobze7ry6szqus5ypn4p.py
# Source Nodes: [x_446], Original ATen: [aten.gelu, aten.gelu_backward]
# x_446 => add_209, erf_23, mul_243
triton_poi_fused_gelu_gelu_backward_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_6', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/5f/c5fpxcqtmpv6hfcvgoggp4uulkvul7omrhuqyeocs2vker7mmf7z.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4096
    x1 = (xindex // 4096)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r2) + (401408*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h5/ch5wdo7kwh5caabsgc4j4viwexl2vygigxh63m6pxmiaru7hon3r.py
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
    size_hints=[4096, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/iw/ciwbq6e2gd4xxq3srhfaj6hlrmwteoc3zsrgxmcspiqstntzf3h3.py
# Source Nodes: [div__44], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__44 => div_68
triton_per_fused_add_div_mul_native_layer_norm_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 392
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
    x3 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 1024.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.8999999985098839
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gy/cgyp5pf3eupbthi7nyfux7wxyymkv5g5jjwtdicmrqrhfmn6fujf.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tj/ctju5vkv6iv3wiikl63j2a42ucyuh64xv2bvautsxmyim6nkpfth.py
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
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 32
    x3 = (xindex // 50176)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (1024*x1) + (50176*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/az/cazzuoyc7cwjtxjmdjmjigo4zacwogtloyqs3xfb5kr3byr2n3gu.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12544
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
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tl.store(out_ptr1 + (r1 + (49*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3g/c3gpg3x6qpftbwibypmnerwnqugmshuvxghaduas4p44f4a4j72t.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]

triton_per_fused__softmax_backward_data_sum_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_sum_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 76832
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (x3 + (76832*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3 + (76832*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x1 + (1568*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 - tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2o/c2ou4n433xxh524vdyeaj6m5g4z2lwjylzpgkmhdrynmgblti6n5.py
# Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]

triton_poi_fused_index_put_new_zeros_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_new_zeros_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7q/c7qwawv4tl57qabirycxkgyxekhxopofno37oielhbb3ces6rmkz.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 37632
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y6 = (yindex // 1568)
    x4 = xindex
    y7 = yindex
    y0 = yindex % 49
    y8 = (yindex // 49)
    y1 = (yindex // 49) % 32
    y2 = (yindex // 1568) % 8
    y3 = (yindex // 12544)
    tmp0 = y6
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + (32*y7)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.1767766952966369
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 16, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr1 + ((-401408) + y0 + (49*x4) + (1568*y8)), tmp13 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp0 >= tmp11
    tmp18 = tl.full([1, 1], 24, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tl.load(in_ptr2 + ((-802816) + x4 + (32*y7)), tmp17 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp17, tmp20, tmp21)
    tmp23 = tl.where(tmp13, tmp16, tmp22)
    tmp24 = tl.where(tmp4, tmp9, tmp23)
    tl.store(out_ptr0 + (x4 + (32*y1) + (1024*y3) + (3072*y0) + (150528*y2)), tmp24, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f6/cf6dq2qiyly2by5swiskhq5g4klfqqiz7yeeyruiloccozkocbqz.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3072
    x1 = (xindex // 3072)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3072*r2) + (301056*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ga/cgangus65dvht4gmqayqzzx4spoz3kond5zrhr7imak5gnb4nhtd.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i7/ci7h7gbw2jmscjren6xdljjtlvgdr4tcnu3ypqy6z3uxyose72js.py
# Source Nodes: [div__43], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__43 => div_66
triton_per_fused_add_div_mul_native_layer_norm_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_18', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 392
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
    x3 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 1024.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.9043478220701218
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pd/cpdtiyurdczi2bbdzh5wyqrhvgctuwh3nsxb5erxysfavsryn7w6.py
# Source Nodes: [shifted_x_88], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# shifted_x_88 => mul_226, sub_70
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_19', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):
    xnumel = 392
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
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
    tmp18 = 1024.0
    tmp19 = tmp10 / tmp18
    tmp20 = tmp2 * tmp18
    tmp21 = tmp20 - tmp6
    tmp22 = tmp11 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp19 * tmp23
    tmp25 = tmp17 + tmp24
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cq/ccqafdklj7ivdskasqljtqw63p7wmmegj75efjiwcsmnojqyhmfd.py
# Source Nodes: [shifted_x_88], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# shifted_x_88 => mul_226, sub_70
triton_red_fused_native_layer_norm_native_layer_norm_backward_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r2 + (98*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r2 + (98*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp5 = tmp3 * tmp4
        tmp6 = tmp0 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
        tmp10 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uu/cuurjv2o4rcqdsnqt3c5ectzmv6xermbite7ag5lgynmrwzn444a.py
# Source Nodes: [], Original ATen: [aten.clone, aten.native_layer_norm_backward]

triton_red_fused_clone_native_layer_norm_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clone_native_layer_norm_backward_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp7 = tmp2 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tmp12 * tmp13
        tmp15 = 2048.0
        tmp16 = tmp14 * tmp15
        tmp17 = tmp16 - tmp4
        tmp19 = tmp18 * tmp9
        tmp20 = tmp17 - tmp19
        tmp21 = tmp11 * tmp20
        tl.store(out_ptr2 + (r1 + (2048*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a7/ca7fcdgwrh4fetpnncmgzduwanrxemashyae5oaujezji2zjf26n.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2) + (200704*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (2048*r2) + (200704*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/by/cbybfwht5nryhduimdyg2acntrcsukyrb4jhpsh67bgofqb65t5n.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_23', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ra/craqxwr737owyrtddaw4ucyqzbrtuhc3rqyjlk2f4hf7g4hlgzbr.py
# Source Nodes: [div__41], Original ATen: [aten.div, aten.mul]
# div__41 => div_63
triton_poi_fused_div_mul_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mul_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 196
    x2 = (xindex // 100352)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*((x1 // 14) % 2)) + (1024*(x1 % 14)) + (14336*(x1 // 28)) + (100352*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp2 = 0.9086956530809402
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/g7/cg7fombludhyduxfmhqvyh7sb5wcj7k35esxgcfg2fojue7myshp.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*r2) + (61952*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r3/cr36uqjdrno7sfdew3aatl5s7hbyj4uzblxmc2566wephmjqistj.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/zd/czdpad6zpnwwudqymi4wzhu3h2r6a5twbyctgrgxydoht2eozif6.py
# Source Nodes: [x_405], Original ATen: [aten.gelu, aten.gelu_backward]
# x_405 => add_191, erf_21, mul_221
triton_poi_fused_gelu_gelu_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
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


# kernel path: /tmp/torchinductor_youkaichao/vs/cvsi6o2nc23rblutnal7p5jd7jmg4cb55x54jckfwmel4fj3ab2e.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 26624
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 2048)
    x0 = xindex % 2048
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (2048*r2) + (247808*x1)), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7f/c7fg2pjvuo6m2j2vps2dudxte22czva4pxh2sodfbon5bq34ahim.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/z6/cz6333kc6p4ojvh364rcg2jv5maxkbigzbnwgoocptwjvakrzghd.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 196
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (r1 + (512*((x2 // 14) % 2)) + (1024*(x2 % 14)) + (14336*(x2 // 28)) + (100352*x3)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dx/cdx6hxpx75j3jrgq72mpiqdlylin3bqxd2xjilsrxiqyio7i7o75.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/jd/cjdqrplb3csvgn6n3q2icmkwrzqjrnvkbrj3g27xxxf74kxzcswa.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 7
    x2 = (xindex // 3584) % 7
    x3 = (xindex // 25088) % 2
    x4 = (xindex // 50176) % 2
    x5 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*((3 + x1 + (7*x3)) % 14)) + (7168*((3 + x2 + (7*x4)) % 14)) + (100352*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x5), None, eviction_policy='evict_last')
    tmp2 = 0.9086956530809402
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4o/c4obehcsogihyio3xlbgmcz3qonpeivywrenu6fzchjl3vdtcorr.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 16
    x3 = (xindex // 25088)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (512*x1) + (25088*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ya/cya57btfeqtiyfaxw4xw3ahccyqdfp3mdxlpir52vynl43s7hecs.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
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
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tl.store(out_ptr1 + (r1 + (49*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/en/cenir4g4frukxrmpgkjy62wrhsgujxh5zpga5otyuvx6e2266536.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[65536, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 38416
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (x3 + (38416*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3 + (38416*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x1 + (784*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 - tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e7/ce7a5ei4m4sgz4iri2gpsamzpz3yswzkzz4gtrv4li5o5qk767uq.py
# Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]

triton_poi_fused_index_put_new_zeros_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_new_zeros_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b7/cb7ln4ic5zosxrcrxz2mj35v3iljn3rbwel7evrygxt4illoj53j.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 75264
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y6 = (yindex // 784)
    x4 = xindex
    y7 = yindex
    y0 = yindex % 49
    y8 = (yindex // 49)
    y1 = (yindex // 49) % 16
    y2 = (yindex // 784) % 32
    y3 = (yindex // 25088)
    tmp0 = y6
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + (32*y7)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.1767766952966369
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 64, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr1 + ((-802816) + y0 + (49*x4) + (1568*y8)), tmp13 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp0 >= tmp11
    tmp18 = tl.full([1, 1], 96, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tl.load(in_ptr2 + ((-1605632) + x4 + (32*y7)), tmp17 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp17, tmp20, tmp21)
    tmp23 = tl.where(tmp13, tmp16, tmp22)
    tmp24 = tl.where(tmp4, tmp9, tmp23)
    tl.store(out_ptr0 + (x4 + (32*y1) + (512*y3) + (1536*y0) + (75264*y2)), tmp24, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xh/cxh733a6ffhccg62uobgu7atvjf7zvgus564huxxvb4wqwd3kixe.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 19968
    rnumel = 121
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
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (1536*r2) + (185856*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5p/c5prk63nfugjc6sjacujxqjcvafy2iicoharfb5qxelpkvzl77ex.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_39', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ec/cec4xttu7ga6nudgzf5uvtla3kbpejpp25mgrstd5wu3hppl3y5g.py
# Source Nodes: [div__39], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
# div__39 => div_60
triton_per_fused_add_div_mul_native_layer_norm_backward_roll_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_roll_40', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (512*(((11 + x0) % 14) % 7)) + (3584*(((11 + x1) % 14) % 7)) + (25088*(((11 + x0) % 14) // 7)) + (50176*(((11 + x1) % 14) // 7)) + (100352*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.9130434766411781
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r3 + (512*x4)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (512*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yl/cyl5id4sdpgbulvvarls5rqmjesm6yczmijiwiqrgkam7fnxveyl.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]

triton_red_fused_native_layer_norm_backward_roll_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_roll_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (512*(((11 + ((r2 + (121*x0)) % 14)) % 14) % 7)) + (3584*(((11 + (((r2 + (121*x0)) // 14) % 14)) % 14) % 7)) + (25088*(((11 + ((r2 + (121*x0)) % 14)) % 14) // 7)) + (50176*(((11 + (((r2 + (121*x0)) // 14) % 14)) % 14) // 7)) + (100352*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (512*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ja/cjaa5dw43546tjpophbm7dzsnjqihekdzom2f6t2zaecdrz5tyil.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]

triton_per_fused_native_layer_norm_backward_roll_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_roll_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
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


# kernel path: /tmp/torchinductor_youkaichao/we/cwea43ngimvgckpb6azcpsfv663hrrhpxqwzftdqocrw7wzjry4x.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_43', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7r/c7rdxzmfp7apfoqk5g2ewvrapslk3ylqlvatj3sqkyhy4ds52cvp.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3584
    x1 = (xindex // 3584) % 7
    x2 = (xindex // 25088) % 2
    x5 = (xindex // 50176)
    x4 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3584*x2) + (7168*x1) + (50176*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = 0.9130434766411781
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6m/c6mo62luzbzw2fdc555ozyjsig4bmfif7wyimhcjxim7so6h64go.py
# Source Nodes: [div__37], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__37 => div_57
triton_per_fused_add_div_mul_native_layer_norm_backward_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_45', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (512*(x0 % 7)) + (3584*(x1 % 7)) + (25088*(x0 // 7)) + (50176*(x1 // 7)) + (100352*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.917391300201416
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r3 + (512*x4)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (512*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/be/cbe3hpq4sjxjcdexjzvyuxkcxaj3btig2rqxxq4xmlgubcpv6wnb.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*(((r2 + (121*x1)) % 14) % 7)) + (3584*((((r2 + (121*x1)) // 14) % 14) % 7)) + (25088*(((r2 + (121*x1)) % 14) // 7)) + (50176*((((r2 + (121*x1)) // 14) % 14) // 7)) + (100352*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/xy/cxyhmxzkgu73cgneallccr7actb7ct42tfgmlmia2avg566pivwu.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 7
    x2 = (xindex // 3584) % 7
    x3 = (xindex // 25088) % 2
    x4 = (xindex // 50176) % 2
    x5 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*((3 + x1 + (7*x3)) % 14)) + (7168*((3 + x2 + (7*x4)) % 14)) + (100352*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x5), None, eviction_policy='evict_last')
    tmp2 = 0.917391300201416
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7h/c7h4nj6bu7gky7trjp7vdraxzbkihwttngssqm6sfqls2vwh5ccr.py
# Source Nodes: [div__35], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
# div__35 => div_54
triton_per_fused_add_div_mul_native_layer_norm_backward_roll_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_roll_48', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (512*(((11 + x0) % 14) % 7)) + (3584*(((11 + x1) % 14) % 7)) + (25088*(((11 + x0) % 14) // 7)) + (50176*(((11 + x1) % 14) // 7)) + (100352*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.9217391312122345
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r3 + (512*x4)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (512*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/in/cin7w5xmnrcwlutsk74pimkznkv3nb4yohdljgdbsu6pyt2whlqf.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3584
    x1 = (xindex // 3584) % 7
    x2 = (xindex // 25088) % 2
    x5 = (xindex // 50176)
    x4 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3584*x2) + (7168*x1) + (50176*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = 0.9217391312122345
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m4/cm4okuhm6k5lyyr36r5laccsgxujk23ejxlg66rnlxfscyddpia3.py
# Source Nodes: [div__33], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__33 => div_51
triton_per_fused_add_div_mul_native_layer_norm_backward_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_50', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (512*(x0 % 7)) + (3584*(x1 % 7)) + (25088*(x0 // 7)) + (50176*(x1 // 7)) + (100352*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.9260869547724724
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r3 + (512*x4)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (512*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ql/cqlwln2ymmvw4l7ctcrdaslloltjl6dwjebxqd3ao2ihehg5ms5u.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 7
    x2 = (xindex // 3584) % 7
    x3 = (xindex // 25088) % 2
    x4 = (xindex // 50176) % 2
    x5 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*((3 + x1 + (7*x3)) % 14)) + (7168*((3 + x2 + (7*x4)) % 14)) + (100352*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x5), None, eviction_policy='evict_last')
    tmp2 = 0.9260869547724724
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vc/cvcssps27lkpgz2yd5fm24x5jvb4de5gria5hrivxanr7ynquvxj.py
# Source Nodes: [div__31], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
# div__31 => div_48
triton_per_fused_add_div_mul_native_layer_norm_backward_roll_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_roll_52', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (512*(((11 + x0) % 14) % 7)) + (3584*(((11 + x1) % 14) % 7)) + (25088*(((11 + x0) % 14) // 7)) + (50176*(((11 + x1) % 14) // 7)) + (100352*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.9304347857832909
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r3 + (512*x4)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (512*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ld/cldov4cdh5sk3wnhkcurf2w5f274biv372hglvzhdavhnlp7idw2.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3584
    x1 = (xindex // 3584) % 7
    x2 = (xindex // 25088) % 2
    x5 = (xindex // 50176)
    x4 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3584*x2) + (7168*x1) + (50176*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = 0.9304347857832909
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/g5/cg5vkyu3gxwhjepeanhckxemct3eveqgtxt5bvpzapjvgfdxp3cp.py
# Source Nodes: [div__29], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__29 => div_45
triton_per_fused_add_div_mul_native_layer_norm_backward_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_54', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (512*(x0 % 7)) + (3584*(x1 % 7)) + (25088*(x0 // 7)) + (50176*(x1 // 7)) + (100352*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.9347826093435287
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r3 + (512*x4)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (512*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gs/cgsaxxtkpxu3krbd2kgj5wzm32f6qz656cubljcnpwzoqwfu6fjc.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 7
    x2 = (xindex // 3584) % 7
    x3 = (xindex // 25088) % 2
    x4 = (xindex // 50176) % 2
    x5 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*((3 + x1 + (7*x3)) % 14)) + (7168*((3 + x2 + (7*x4)) % 14)) + (100352*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x5), None, eviction_policy='evict_last')
    tmp2 = 0.9347826093435287
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qn/cqnzrtv4ywevwzfgdmjyecdgx6fkrb7jftsiek7enhonnssvglme.py
# Source Nodes: [div__27], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
# div__27 => div_42
triton_per_fused_add_div_mul_native_layer_norm_backward_roll_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_roll_56', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (512*(((11 + x0) % 14) % 7)) + (3584*(((11 + x1) % 14) % 7)) + (25088*(((11 + x0) % 14) // 7)) + (50176*(((11 + x1) % 14) // 7)) + (100352*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.9391304366290569
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r3 + (512*x4)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (512*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/td/ctdehg2mcz5pz5mya4psjakis2ofgevsbzrvu55u3tw4by4l7iod.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3584
    x1 = (xindex // 3584) % 7
    x2 = (xindex // 25088) % 2
    x5 = (xindex // 50176)
    x4 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3584*x2) + (7168*x1) + (50176*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = 0.9391304366290569
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fn/cfn6655x4znktrxpi6rhksj5vbzzxtgsv3irxum7chyxfiwecbag.py
# Source Nodes: [div__25], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__25 => div_39
triton_per_fused_add_div_mul_native_layer_norm_backward_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_58', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (512*(x0 % 7)) + (3584*(x1 % 7)) + (25088*(x0 // 7)) + (50176*(x1 // 7)) + (100352*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.9434782639145851
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r3 + (512*x4)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (512*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wg/cwgb7vn2uknrl2wibrsrk5zmy3grjc7uwyd37ewxe3algotwa5sl.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 7
    x2 = (xindex // 3584) % 7
    x3 = (xindex // 25088) % 2
    x4 = (xindex // 50176) % 2
    x5 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*((3 + x1 + (7*x3)) % 14)) + (7168*((3 + x2 + (7*x4)) % 14)) + (100352*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x5), None, eviction_policy='evict_last')
    tmp2 = 0.9434782639145851
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mc/cmcznldj5erwimxhcyrcghmajrzfw46ktqcomdzxyttesicsz4es.py
# Source Nodes: [div__23], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
# div__23 => div_36
triton_per_fused_add_div_mul_native_layer_norm_backward_roll_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_roll_60', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (512*(((11 + x0) % 14) % 7)) + (3584*(((11 + x1) % 14) % 7)) + (25088*(((11 + x0) % 14) // 7)) + (50176*(((11 + x1) % 14) // 7)) + (100352*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.947826087474823
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r3 + (512*x4)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (512*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bz/cbzsddaokixjqvf5cu2bsz2md4kweutupqumcbmnmx3q6tdmvh54.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_61', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3584
    x1 = (xindex // 3584) % 7
    x2 = (xindex // 25088) % 2
    x5 = (xindex // 50176)
    x4 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3584*x2) + (7168*x1) + (50176*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = 0.947826087474823
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/br/cbrejniu5gqmxh7ogluqx474csj7jfxwbvbkiesupl3bibvduy7m.py
# Source Nodes: [div__21], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__21 => div_33
triton_per_fused_add_div_mul_native_layer_norm_backward_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_62', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (512*(x0 % 7)) + (3584*(x1 % 7)) + (25088*(x0 // 7)) + (50176*(x1 // 7)) + (100352*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.9521739110350609
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r3 + (512*x4)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (512*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bl/cblujrizsu6vrgx2gpix5whj53b54voliyptt3u3iqtvdfz5umnc.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 7
    x2 = (xindex // 3584) % 7
    x3 = (xindex // 25088) % 2
    x4 = (xindex // 50176) % 2
    x5 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*((3 + x1 + (7*x3)) % 14)) + (7168*((3 + x2 + (7*x4)) % 14)) + (100352*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x5), None, eviction_policy='evict_last')
    tmp2 = 0.9521739110350609
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bp/cbpdluggp6rrx6pc75qfemxkhg5hh6xtpvgpfdu3y27pcph2r4cc.py
# Source Nodes: [div__19], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
# div__19 => div_30
triton_per_fused_add_div_mul_native_layer_norm_backward_roll_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_roll_64', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (512*(((11 + x0) % 14) % 7)) + (3584*(((11 + x1) % 14) % 7)) + (25088*(((11 + x0) % 14) // 7)) + (50176*(((11 + x1) % 14) // 7)) + (100352*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.9565217345952988
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r3 + (512*x4)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (512*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x2/cx2evnkybazvvgsrh7ucxq5usu6uat4ehfxh6rfdt7r2p7zj6q4v.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3584
    x1 = (xindex // 3584) % 7
    x2 = (xindex // 25088) % 2
    x5 = (xindex // 50176)
    x4 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3584*x2) + (7168*x1) + (50176*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = 0.9565217345952988
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bu/cbuqqg4bjgsziewjl3drbo2o3ew3xjimk43di4mbsvgv4ky7q7nj.py
# Source Nodes: [div__17], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__17 => div_27
triton_per_fused_add_div_mul_native_layer_norm_backward_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_66', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (512*(x0 % 7)) + (3584*(x1 % 7)) + (25088*(x0 // 7)) + (50176*(x1 // 7)) + (100352*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.960869561880827
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r3 + (512*x4)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (512*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hx/chxrn5wjuiejnvrtttp4cw6vzjwrteygasedorcqh77iqfnpejwx.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 7
    x2 = (xindex // 3584) % 7
    x3 = (xindex // 25088) % 2
    x4 = (xindex // 50176) % 2
    x5 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*((3 + x1 + (7*x3)) % 14)) + (7168*((3 + x2 + (7*x4)) % 14)) + (100352*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x5), None, eviction_policy='evict_last')
    tmp2 = 0.960869561880827
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bz/cbzhpugark75tpvgkb3eihni4xt4kfmueeawthbfty7l7ju256kk.py
# Source Nodes: [div__15], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
# div__15 => div_24
triton_per_fused_add_div_mul_native_layer_norm_backward_roll_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_roll_68', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (512*(((11 + x0) % 14) % 7)) + (3584*(((11 + x1) % 14) % 7)) + (25088*(((11 + x0) % 14) // 7)) + (50176*(((11 + x1) % 14) // 7)) + (100352*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.9652173891663551
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r3 + (512*x4)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (512*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eh/cehjoymeajba3bohonzxjpy2ubbuonqlj2joldod4gwhecww3aeb.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3584
    x1 = (xindex // 3584) % 7
    x2 = (xindex // 25088) % 2
    x5 = (xindex // 50176)
    x4 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3584*x2) + (7168*x1) + (50176*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = 0.9652173891663551
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/w4/cw4vmkqi5o6zdvbmftgmah63pxi54q2kc5drjulem6hif2g2j7ku.py
# Source Nodes: [div__13], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__13 => div_21
triton_per_fused_add_div_mul_native_layer_norm_backward_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_70', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (512*(x0 % 7)) + (3584*(x1 % 7)) + (25088*(x0 // 7)) + (50176*(x1 // 7)) + (100352*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.9695652164518833
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r3 + (512*x4)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (512*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6l/c6lonwii5iiadbpxqjho4tv5ltrqme5sqjesx65mkacyjtftoi4r.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 7
    x2 = (xindex // 3584) % 7
    x3 = (xindex // 25088) % 2
    x4 = (xindex // 50176) % 2
    x5 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*((3 + x1 + (7*x3)) % 14)) + (7168*((3 + x2 + (7*x4)) % 14)) + (100352*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x5), None, eviction_policy='evict_last')
    tmp2 = 0.9695652164518833
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6e/c6e3gpn7xfib5mnuek5etmomh2d3dux6qernt6b4rwf5byj7sr4w.py
# Source Nodes: [div__11], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
# div__11 => div_18
triton_per_fused_add_div_mul_native_layer_norm_backward_roll_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_roll_72', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (512*(((11 + x0) % 14) % 7)) + (3584*(((11 + x1) % 14) % 7)) + (25088*(((11 + x0) % 14) // 7)) + (50176*(((11 + x1) % 14) // 7)) + (100352*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.9739130418747663
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r3 + (512*x4)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (512*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aq/caqzp35m2v4mwp6z5ljhl5fikhvepk4x3h66thv4fkcwjqn2we4x.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3584
    x1 = (xindex // 3584) % 7
    x2 = (xindex // 25088) % 2
    x5 = (xindex // 50176)
    x4 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3584*x2) + (7168*x1) + (50176*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = 0.9739130418747663
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bl/cblaqghhskx7w46frl7byevrzbmau4p5l5zblwxdd62geqr3if3n.py
# Source Nodes: [div__9], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__9 => div_15
triton_per_fused_add_div_mul_native_layer_norm_backward_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_74', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (512*(x0 % 7)) + (3584*(x1 % 7)) + (25088*(x0 // 7)) + (50176*(x1 // 7)) + (100352*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.9782608672976494
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r3 + (512*x4)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (512*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nm/cnm5jjmh5v3wwfdv5hyxf6p6mzb3fapeenyxire3nnk2qez2zkxa.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 7
    x2 = (xindex // 3584) % 7
    x3 = (xindex // 25088) % 2
    x4 = (xindex // 50176) % 2
    x5 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*((3 + x1 + (7*x3)) % 14)) + (7168*((3 + x2 + (7*x4)) % 14)) + (100352*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x5), None, eviction_policy='evict_last')
    tmp2 = 0.9782608672976494
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/w6/cw6e5vws46tcijvgglimfxqdq5q5jn4xrvtlvwbx4ohesmqj5guo.py
# Source Nodes: [div__7], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
# div__7 => div_12
triton_per_fused_add_div_mul_native_layer_norm_backward_roll_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_roll_76', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (512*(((11 + x0) % 14) % 7)) + (3584*(((11 + x1) % 14) % 7)) + (25088*(((11 + x0) % 14) // 7)) + (50176*(((11 + x1) % 14) // 7)) + (100352*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.9826086945831776
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r3 + (512*x4)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (512*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n5/cn5ulqsj3ocjy335xax6vl4ydcyfn43d52prkjwkrpdita2k4g73.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3584
    x1 = (xindex // 3584) % 7
    x2 = (xindex // 25088) % 2
    x5 = (xindex // 50176)
    x4 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3584*x2) + (7168*x1) + (50176*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = 0.9826086945831776
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q4/cq47qvqqs2bkugcfhpw4tlms3kqrwirofdopl4nuc6h55xpaxnxf.py
# Source Nodes: [shifted_x_16], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# shifted_x_16 => mul_44, sub_15
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_78', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (512*(x0 % 7)) + (3584*(x1 % 7)) + (25088*(x0 // 7)) + (50176*(x1 // 7)) + (100352*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x4), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_out_ptr0 + (r3 + (512*x4)), rmask & xmask, other=0.0)
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
    tmp18 = 512.0
    tmp19 = tmp10 / tmp18
    tmp20 = tmp2 * tmp18
    tmp21 = tmp20 - tmp6
    tmp22 = tmp11 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp19 * tmp23
    tmp25 = tmp17 + tmp24
    tl.store(in_out_ptr0 + (r3 + (512*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xa/cxasp7ektj64f4vtlozvwi7ppirxz6z57gxzn5x4k43doaooljq2.py
# Source Nodes: [shifted_x_16], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# shifted_x_16 => mul_44, sub_15
triton_red_fused_native_layer_norm_native_layer_norm_backward_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_79', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
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
        tmp3 = tl.load(in_ptr0 + (x0 + (512*(((r2 + (121*x1)) % 14) % 7)) + (3584*((((r2 + (121*x1)) // 14) % 14) % 7)) + (25088*(((r2 + (121*x1)) % 14) // 7)) + (50176*((((r2 + (121*x1)) // 14) % 14) // 7)) + (100352*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qi/cqicfyb7ceb6rqol6v7jhkrqzigzb2ykeugnbdvarqk6yu7wppzg.py
# Source Nodes: [], Original ATen: [aten.clone, aten.native_layer_norm_backward]

triton_per_fused_clone_native_layer_norm_backward_80 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_native_layer_norm_backward_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 1024.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ah/cahu2bhibg3iyrhthtxtvorvdzsy5g44olhfpi72tjfb5mog46ry.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 13312
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 1024)
    x0 = xindex % 1024
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (1024*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (1024*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/wg/cwgdzcgoekmyisksi7q4d6itzhaqvfbz5subfuow3qlxr2fffzx7.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_82 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_82', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e7/ce7s2qxfcqwcg62sjqgbvcqpmp4voasvmtywcix2d5i2ckkwqmqx.py
# Source Nodes: [div__5], Original ATen: [aten.div, aten.mul]
# div__5 => div_9
triton_poi_fused_div_mul_83 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mul_83', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 784
    x2 = (xindex // 200704)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*((x1 // 28) % 2)) + (512*(x1 % 28)) + (14336*(x1 // 56)) + (200704*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp2 = 0.9869565209373832
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/iu/ciue6p6i6degym7blvmrdk2yaakp5duq43phinalt66olab33img.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_84', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4a/c4ayyqicpsm4zzeklmmutjncjtkqtanedt6dmalskbwrxlivbpwv.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_85 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uf/cufcbnijxhk6fepea6h4rrygdynxl5s5zgsvz6uwwwnuwe54omfa.py
# Source Nodes: [x_76], Original ATen: [aten.gelu, aten.gelu_backward]
# x_76 => add_36, erf_3, mul_39
triton_poi_fused_gelu_gelu_backward_86 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_86', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
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


# kernel path: /tmp/torchinductor_youkaichao/oj/cojjf5odqr7da2d2nxczlocqhbjxnjj2fp2gb4tfrpdtlvjufpvm.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ah/cahave6yqkhfvf5twohuilhlbr7f63kbhvfvtoftma6c5ixhirr7.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_88 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_88', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 49
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/fk/cfkzgfzl7ou67jbktfrycozqa727taccl4fz7df6gy7uxl7bukxk.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_89 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 6272
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
    x2 = xindex % 784
    x3 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (r1 + (256*((x2 // 28) % 2)) + (512*(x2 % 28)) + (14336*(x2 // 56)) + (200704*x3)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 256.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nw/cnwqsfmb7dj6toa5x6qz6z5znguwssqpbzchwc4mwtdn7ymcjn2g.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_90', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/7r/c7rylpdhj2ycc2kcqivnb3y6jkgwivutexwzadq5lbaunmsys4ll.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_91 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_91', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 7
    x2 = (xindex // 1792) % 7
    x3 = (xindex // 12544) % 4
    x4 = (xindex // 50176) % 4
    x5 = (xindex // 200704)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*((3 + x1 + (7*x3)) % 28)) + (7168*((3 + x2 + (7*x4)) % 28)) + (200704*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x5), None, eviction_policy='evict_last')
    tmp2 = 0.9869565209373832
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zo/czoluitovmbsz2swfzuxgruec4ucgqtaikbyte4vnnzecez4w6s7.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_92', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 8
    x3 = (xindex // 12544)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (256*x1) + (12544*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sy/csyhvngpujf3gs4lpgb3aw4yzrjilia5cymlspiysivskkr6z4mo.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_93 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[65536, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_93', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
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
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tl.store(out_ptr1 + (r1 + (49*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pu/cpupvbnntoiynqwn77e3ouy6huuror4qtf6n5bmcp3odqhff6hhq.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_94 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_94', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 19208
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = (xindex // 49)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x3 + (19208*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x3 + (19208*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x1 + (392*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp1 * tmp3
        tmp5 = tmp2 - tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/23/c235j6d6bae6lu7lknfli4laeb7pagcknl7xdtvgdvwhsh2japtc.py
# Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]

triton_poi_fused_index_put_new_zeros_95 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_new_zeros_95', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rs/crsgesxqp6j4xgt5sznm6dq3fkesxwutiirnlgz37536qekkin3g.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_96 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_96', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 150528
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y6 = (yindex // 392)
    x4 = xindex
    y7 = yindex
    y0 = yindex % 49
    y8 = (yindex // 49)
    y1 = (yindex // 49) % 8
    y2 = (yindex // 392) % 128
    y3 = (yindex // 50176)
    tmp0 = y6
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + (32*y7)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.1767766952966369
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 256, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr1 + ((-1605632) + y0 + (49*x4) + (1568*y8)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp0 >= tmp11
    tmp18 = tl.full([1, 1], 384, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tl.load(in_ptr2 + ((-3211264) + x4 + (32*y7)), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp17, tmp20, tmp21)
    tmp23 = tl.where(tmp13, tmp16, tmp22)
    tmp24 = tl.where(tmp4, tmp9, tmp23)
    tl.store(out_ptr0 + (x4 + (32*y1) + (256*y3) + (768*y0) + (37632*y2)), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fo/cfoh47waztb6kveofgmy27fm2w56ndttsfd32llyuatjzfp4yhfi.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_97 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_97', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kx/ckxfs26ohvsh5c5ucs6dnnvypfzvdhaifyyujs2zxox5klz6spgv.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_98 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_98', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gr/cgr7xnk7npgwnfga2yegyhdxmwxicq6f4npe7kvzw5yt3mf55md4.py
# Source Nodes: [div__3], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
# div__3 => div_6
triton_per_fused_add_div_mul_native_layer_norm_backward_roll_99 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_roll_99', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 28
    x1 = (xindex // 28) % 28
    x2 = (xindex // 784)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (256*(((25 + x0) % 28) % 7)) + (1792*(((25 + x1) % 28) % 7)) + (12544*(((25 + x0) % 28) // 7)) + (50176*(((25 + x1) % 28) // 7)) + (200704*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (256*x4)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r3 + (256*x4)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 256.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = 0.9913043472915888
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r3 + (256*x4)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (256*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kn/cknoy7iqnn5465ocyelfi374awhafx6urh44jmbpk22uo7r26ogj.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]

triton_red_fused_native_layer_norm_backward_roll_100 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_roll_100', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (256*(((25 + ((r2 + (128*x0)) % 28)) % 28) % 7)) + (1792*(((25 + (((r2 + (128*x0)) // 28) % 28)) % 28) % 7)) + (12544*(((25 + ((r2 + (128*x0)) % 28)) % 28) // 7)) + (50176*(((25 + (((r2 + (128*x0)) // 28) % 28)) % 28) // 7)) + (200704*((r2 + (128*x0)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/fp/cfpedllmes2f6z6qcsrtolqcy2kk2lj46osjayswtrmgrlmvmqra.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]

triton_per_fused_native_layer_norm_backward_roll_101 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_roll_101', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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


# kernel path: /tmp/torchinductor_youkaichao/ok/cokfijuk5gwvap7t56ypeeehd2q56pae7zp2t276bzy7wb7fvbdd.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_102 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_102', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 6272
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 256.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fs/cfsslykf6tmzlutjivbxcnbnpwzwjfebwl7zflzmbbw4v4ijsrhn.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_103 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_103', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1792
    x1 = (xindex // 1792) % 7
    x2 = (xindex // 12544) % 4
    x5 = (xindex // 50176)
    x4 = (xindex // 200704)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1792*x2) + (7168*x1) + (50176*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = 0.9913043472915888
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hl/chlefp2ckqn64pe4rmscoif7lbx6ulunjlmnyxqg6w3p3khmwe4y.py
# Source Nodes: [shifted_x_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# shifted_x_8 => mul_22, sub_8
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_104 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_104', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 28
    x1 = (xindex // 28) % 28
    x2 = (xindex // 784)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (256*(x0 % 7)) + (1792*(x1 % 7)) + (12544*(x0 // 7)) + (50176*(x1 // 7)) + (200704*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r3 + (256*x4)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x4), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_out_ptr0 + (r3 + (256*x4)), rmask & xmask, other=0.0)
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
    tmp18 = 256.0
    tmp19 = tmp10 / tmp18
    tmp20 = tmp2 * tmp18
    tmp21 = tmp20 - tmp6
    tmp22 = tmp11 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp19 * tmp23
    tmp25 = tmp17 + tmp24
    tl.store(in_out_ptr0 + (r3 + (256*x4)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pk/cpkf7nnrcqc7nzakjforgs63ux7pjmxrb3selhqki6fhwnuiqekg.py
# Source Nodes: [shifted_x_8], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# shifted_x_8 => mul_22, sub_8
triton_red_fused_native_layer_norm_native_layer_norm_backward_105 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_105', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*(((r2 + (128*x1)) % 28) % 7)) + (1792*((((r2 + (128*x1)) // 28) % 28) % 7)) + (12544*(((r2 + (128*x1)) % 28) // 7)) + (50176*((((r2 + (128*x1)) // 28) % 28) // 7)) + (200704*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r2 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r2 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp5 = tmp3 * tmp4
        tmp6 = tmp0 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp10 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n6/cn6cgcp3kf2ylowez7jwh5vvk53yrsawvp3zubymfefgx6ch6zyh.py
# Source Nodes: [], Original ATen: [aten.clone, aten.native_layer_norm_backward]

triton_per_fused_clone_native_layer_norm_backward_106 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_native_layer_norm_backward_106', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 512.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rq/crqa3dstwvcqplejjkcwscqnmwyxblr3ugpmqyvwbn4w2lila25f.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_107 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_107', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/vq/cvqw7olfaw4dv223ymw74i2nixk2gnbkw2axf4ngrbmwjuajeucr.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_108 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_108', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qn/cqnbcncgjxtpuifkkdll7wyeeotrg7ozcjwaw5gtvvkmx3l6pnqj.py
# Source Nodes: [div__1], Original ATen: [aten.div, aten.mul]
# div__1 => div_3
triton_poi_fused_div_mul_109 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mul_109', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 3136
    x2 = (xindex // 401408)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*((x1 // 56) % 2)) + (256*(x1 % 56)) + (14336*(x1 // 112)) + (401408*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp2 = 0.9956521736457944
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4e/c4e7iuv5dnr52lnvgp43dt3pg7kh4cuohy2pflm25dvhmmiwyocq.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_110', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fp/cfpu2clceibkdgkzo5uuqh5jya3yfe4a6hvksiois2l6y3sgu3di.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_111 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_111', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bn/cbn7hrvpwmn4wqa4hsuwhn3d52u5mmddu4fec4i37bimcngq6yhg.py
# Source Nodes: [x_35], Original ATen: [aten.gelu, aten.gelu_backward]
# x_35 => add_17, erf_1, mul_17
triton_poi_fused_gelu_gelu_backward_112 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_112', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
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


# kernel path: /tmp/torchinductor_youkaichao/kl/cklyztqwyyd4qrtqgnqxuo6awro2gg4ygoh2fw6ir4b4rx772f7p.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_113 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_113', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6c/c6cuu3mzejrmxtthb6msynlivaufasxni3ezf2mvskwu7xqczcjh.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_114 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_114', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mb/cmbm2qprkfsgukbzo5rzj4cgtkoomvfwwfxtidgcrenmg5o4z5dc.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_115 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_115', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 3136
    x3 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (r1 + (128*((x2 // 56) % 2)) + (256*(x2 % 56)) + (14336*(x2 // 112)) + (401408*x3)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = 128.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/em/cem6np75flgvtw6hfhqqumbogmkgv4r3kw32srquzxo6x6ul6vdu.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_116 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_116', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/pv/cpvh2v2nt3g5xkeun2gvxeta4up7fbzpmmmovmuuob2lddiwz3as.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_117 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_117', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 7
    x2 = (xindex // 896) % 7
    x3 = (xindex // 6272) % 8
    x4 = (xindex // 50176) % 8
    x5 = (xindex // 401408)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*((3 + x1 + (7*x3)) % 56)) + (7168*((3 + x2 + (7*x4)) % 56)) + (401408*x5)), None)
    tmp1 = tl.load(in_ptr1 + (x5), None, eviction_policy='evict_last')
    tmp2 = 0.9956521736457944
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/w4/cw4hr23gst65663kakqbxjjwjfpg3l3kgylydirmz7dvnixo2nw4.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_118 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_118', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 4
    x3 = (xindex // 6272)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (128*x1) + (6272*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wa/cwa6s2hg5nhrtw6vb4he34tpnmzy5cdobtpkzhpjskksgfpnfdnc.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_119 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_119', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tl.store(out_ptr1 + (r1 + (49*x0)), tmp8, rmask)
    tl.store(out_ptr0 + (x0), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cl/ccla6cv2zyfibdcfzygbpxkwli6bmminqlgwq3pcrep5ubj4hids.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_120 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_120', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9604
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = (xindex // 49)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x3 + (9604*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x3 + (9604*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x1 + (196*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp1 * tmp3
        tmp5 = tmp2 - tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6m/c6mmdmqywr74ace4mgdwgno7ijkfj5tifecw5ceujfcw2rciggus.py
# Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]

triton_poi_fused_index_put_new_zeros_121 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_new_zeros_121', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 676
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/54/c543xwvshmvs52kctuo57vsxw7ndjnbavqckqdyitisvzugogfzq.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_122 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_122', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 301056
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y6 = (yindex // 196)
    x4 = xindex
    y7 = yindex
    y0 = yindex % 49
    y8 = (yindex // 49)
    y1 = (yindex // 49) % 4
    y2 = (yindex // 196) % 512
    y3 = (yindex // 100352)
    tmp0 = y6
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + (32*y7)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.1767766952966369
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 1024, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr1 + ((-3211264) + y0 + (49*x4) + (1568*y8)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp0 >= tmp11
    tmp18 = tl.full([1, 1], 1536, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tl.load(in_ptr2 + ((-6422528) + x4 + (32*y7)), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp17, tmp20, tmp21)
    tmp23 = tl.where(tmp13, tmp16, tmp22)
    tmp24 = tl.where(tmp4, tmp9, tmp23)
    tl.store(out_ptr0 + (x4 + (32*y1) + (128*y3) + (384*y0) + (18816*y2)), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uu/cuuksa6arrserdleen7v7hdz2upzsnjanagqqsyfydhkpq3fktwn.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_123 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_123', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 75264
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wp/cwptian56ntsvechqzfo5y476df2wdmd22p5csrha74hsq4abq74.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_124 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_124', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
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
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pq/cpq3lvxim5m7qypl2kgrsgnn2i6cmo4syp6ljbnw4sm65roqn4bk.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.roll]

triton_red_fused_add_native_layer_norm_backward_roll_125 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_roll_125', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 56
    x1 = (xindex // 56) % 56
    x2 = (xindex // 3136)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*(((53 + x0) % 56) % 7)) + (896*(((53 + x1) % 56) % 7)) + (6272*(((53 + x0) % 56) // 7)) + (50176*(((53 + x1) % 56) // 7)) + (401408*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp7 = tmp2 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp12 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp11 = tl.load(in_out_ptr0 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr0 + (r3 + (128*(((53 + x0) % 56) % 7)) + (896*(((53 + x1) % 56) % 7)) + (6272*(((53 + x0) % 56) // 7)) + (50176*(((53 + x1) % 56) // 7)) + (401408*x2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr2 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tmp13 * tmp14
        tmp16 = 128.0
        tmp17 = tmp15 * tmp16
        tmp18 = tmp17 - tmp4
        tmp20 = tmp19 * tmp9
        tmp21 = tmp18 - tmp20
        tmp22 = tmp12 * tmp21
        tmp23 = tmp11 + tmp22
        tl.store(in_out_ptr0 + (r3 + (128*x4)), tmp23, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ps/cpsuysrnhxhaztlveazcyavtddefrj6b3fskneay3o32uumwxux4.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]

triton_red_fused_native_layer_norm_backward_roll_126 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_roll_126', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*(((53 + ((r2 + (128*x0)) % 56)) % 56) % 7)) + (896*(((53 + (((r2 + (128*x0)) // 56) % 56)) % 56) % 7)) + (6272*(((53 + ((r2 + (128*x0)) % 56)) % 56) // 7)) + (50176*(((53 + (((r2 + (128*x0)) // 56) % 56)) % 56) // 7)) + (401408*((r2 + (128*x0)) // 3136))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/mx/cmxb6wh4gtxnudrt5q5hffwxjav7o3vwgccraqrs3hqqtnlyw4zc.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]

triton_per_fused_native_layer_norm_backward_roll_127 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_roll_127', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
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


# kernel path: /tmp/torchinductor_youkaichao/fh/cfhnvm64yyc2vptptmb6vfdvm3xvp6tk3asdmcnzucfjuedup5mf.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_128 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_128', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y3/cy3cakp7nijptip2prcy6vcd3uht4ur4wrh4r2sfhkoaknafujwy.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_129 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_129', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = 128.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/22/c22egxfctjvtx7osou5pafpopyq5nzrpntngw2frpn2omjvxcta4.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_130 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_130', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*((x1 % 49) % 7)) + (896*((x1 // 49) % 8)) + (7168*((x1 % 49) // 7)) + (50176*(x1 // 392))), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tw/ctworlzqjcnwmg7voaik3bh3sm7mgsizjpcittfim3idc2xvchdi.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_131 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_131', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 56
    x1 = (xindex // 56) % 56
    x2 = (xindex // 3136)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*(x0 % 7)) + (896*(x1 % 7)) + (6272*(x0 // 7)) + (50176*(x1 // 7)) + (401408*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp7 = tmp2 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp12 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp11 = tl.load(in_out_ptr0 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr0 + (r3 + (128*(x0 % 7)) + (896*(x1 % 7)) + (6272*(x0 // 7)) + (50176*(x1 // 7)) + (401408*x2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr2 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 * tmp14
        tmp16 = 128.0
        tmp17 = tmp15 * tmp16
        tmp18 = tmp17 - tmp4
        tmp20 = tmp19 * tmp9
        tmp21 = tmp18 - tmp20
        tmp22 = tmp12 * tmp21
        tmp23 = tmp11 + tmp22
        tmp25 = tmp23 * tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
        tl.store(in_out_ptr0 + (r3 + (128*x4)), tmp23, rmask & xmask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    _tmp35 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr5 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tmp29 * tmp30
        tmp33 = tmp31 * tmp32
        tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
        tmp36 = _tmp35 + tmp34
        _tmp35 = tl.where(rmask & xmask, tmp36, _tmp35)
    tmp35 = tl.sum(_tmp35, 1)[:, None]
    tmp37 = tl.load(in_ptr6 + (x4), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp38 = tl.load(in_out_ptr0 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp39 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp44 = tl.load(in_ptr5 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp40 = tmp38 * tmp39
        tmp41 = 128.0
        tmp42 = tmp40 * tmp41
        tmp43 = tmp42 - tmp27
        tmp45 = tmp44 * tmp35
        tmp46 = tmp43 - tmp45
        tmp47 = tmp37 * tmp46
        tl.store(out_ptr4 + (r3 + (128*x4)), tmp47, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jp/cjp2carv4oycixdl372hxhuxt3xlo464p7nb2lxg3a7dkh3tpr3d.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_132 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_132', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*(((r2 + (128*x1)) % 56) % 7)) + (896*((((r2 + (128*x1)) // 56) % 56) % 7)) + (6272*(((r2 + (128*x1)) % 56) // 7)) + (50176*((((r2 + (128*x1)) // 56) % 56) // 7)) + (401408*((r2 + (128*x1)) // 3136))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_25, primals_27, primals_29, primals_35, primals_41, primals_47, primals_53, primals_56, primals_62, primals_68, primals_74, primals_80, primals_83, primals_89, primals_95, primals_101, primals_107, primals_113, primals_119, primals_125, primals_131, primals_137, primals_143, primals_149, primals_155, primals_161, primals_167, primals_173, primals_179, primals_185, primals_191, primals_197, primals_203, primals_209, primals_215, primals_221, primals_227, primals_233, primals_239, primals_245, primals_251, primals_257, primals_263, primals_269, primals_275, primals_281, primals_287, primals_293, primals_299, primals_302, primals_308, primals_314, primals_320, primals_326, primals_365, mul, mul_2, view_3, view_9, view_15, mul_5, view_21, addmm_2, view_23, mul_10, view_29, view_35, view_43, bernoulli, mul_14, view_49, addmm_6, view_51, bernoulli_1, mul_20, view_56, mm, getitem_19, rsqrt_6, view_61, view_67, view_73, bernoulli_2, mul_26, view_79, addmm_10, view_81, bernoulli_3, mul_32, view_87, view_93, view_101, bernoulli_4, mul_36, view_107, addmm_14, view_109, bernoulli_5, mul_42, view_114, mm_1, getitem_35, rsqrt_11, view_119, view_125, view_131, bernoulli_6, mul_48, view_137, addmm_18, view_139, bernoulli_7, mul_54, view_145, view_151, view_159, bernoulli_8, mul_58, view_165, addmm_22, view_167, bernoulli_9, mul_64, view_173, view_179, view_185, bernoulli_10, mul_68, view_191, addmm_26, view_193, bernoulli_11, mul_74, view_199, view_205, view_213, bernoulli_12, mul_78, view_219, addmm_30, view_221, bernoulli_13, mul_84, view_227, view_233, view_239, bernoulli_14, mul_88, view_245, addmm_34, view_247, bernoulli_15, mul_94, view_253, view_259, view_267, bernoulli_16, mul_98, view_273, addmm_38, view_275, bernoulli_17, mul_104, view_281, view_287, view_293, bernoulli_18, mul_108, view_299, addmm_42, view_301, bernoulli_19, mul_114, view_307, view_313, view_321, bernoulli_20, mul_118, view_327, addmm_46, view_329, bernoulli_21, mul_124, view_335, view_341, view_347, bernoulli_22, mul_128, view_353, addmm_50, view_355, bernoulli_23, mul_134, view_361, view_367, view_375, bernoulli_24, mul_138, view_381, addmm_54, view_383, bernoulli_25, mul_144, view_389, view_395, view_401, bernoulli_26, mul_148, view_407, addmm_58, view_409, bernoulli_27, mul_154, view_415, view_421, view_429, bernoulli_28, mul_158, view_435, addmm_62, view_437, bernoulli_29, mul_164, view_443, view_449, view_455, bernoulli_30, mul_168, view_461, addmm_66, view_463, bernoulli_31, mul_174, view_469, view_475, view_483, bernoulli_32, mul_178, view_489, addmm_70, view_491, bernoulli_33, mul_184, view_497, view_503, view_509, bernoulli_34, mul_188, view_515, addmm_74, view_517, bernoulli_35, mul_194, view_523, view_529, view_537, bernoulli_36, mul_198, view_543, addmm_78, view_545, bernoulli_37, mul_204, view_551, view_557, view_563, bernoulli_38, mul_208, view_569, addmm_82, view_571, bernoulli_39, mul_214, view_577, view_583, view_591, bernoulli_40, mul_218, view_597, addmm_86, view_599, bernoulli_41, mul_224, view_604, mm_2, getitem_163, rsqrt_48, view_609, view_615, view_621, bernoulli_42, mul_230, view_627, addmm_90, view_629, bernoulli_43, mul_236, view_635, view_641, view_647, bernoulli_44, mul_240, view_653, addmm_94, view_655, bernoulli_45, mul_246, clone_264, permute_248, div_71, permute_252, permute_256, div_72, permute_261, permute_266, permute_267, alias_24, permute_269, permute_270, permute_273, div_73, permute_278, permute_282, div_74, permute_287, permute_292, permute_293, alias_25, permute_295, permute_296, permute_299, permute_306, div_76, permute_309, permute_313, div_77, permute_318, permute_323, permute_324, alias_26, permute_326, permute_327, permute_330, div_78, permute_335, permute_339, div_79, permute_344, permute_349, permute_350, alias_27, permute_352, permute_353, permute_356, div_80, permute_361, permute_365, div_81, permute_370, permute_375, permute_376, alias_28, permute_378, permute_379, permute_382, div_82, permute_387, permute_391, div_83, permute_396, permute_401, permute_402, alias_29, permute_404, permute_405, permute_408, div_84, permute_413, permute_417, div_85, permute_422, permute_427, permute_428, alias_30, permute_430, permute_431, permute_434, div_86, permute_439, permute_443, div_87, permute_448, permute_453, permute_454, alias_31, permute_456, permute_457, permute_460, div_88, permute_465, permute_469, div_89, permute_474, permute_479, permute_480, alias_32, permute_482, permute_483, permute_486, div_90, permute_491, permute_495, div_91, permute_500, permute_505, permute_506, alias_33, permute_508, permute_509, permute_512, div_92, permute_517, permute_521, div_93, permute_526, permute_531, permute_532, alias_34, permute_534, permute_535, permute_538, div_94, permute_543, permute_547, div_95, permute_552, permute_557, permute_558, alias_35, permute_560, permute_561, permute_564, div_96, permute_569, permute_573, div_97, permute_578, permute_583, permute_584, alias_36, permute_586, permute_587, permute_590, div_98, permute_595, permute_599, div_99, permute_604, permute_609, permute_610, alias_37, permute_612, permute_613, permute_616, div_100, permute_621, permute_625, div_101, permute_630, permute_635, permute_636, alias_38, permute_638, permute_639, permute_642, div_102, permute_647, permute_651, div_103, permute_656, permute_661, permute_662, alias_39, permute_664, permute_665, permute_668, div_104, permute_673, permute_677, div_105, permute_682, permute_687, permute_688, alias_40, permute_690, permute_691, permute_694, div_106, permute_699, permute_703, div_107, permute_708, permute_713, permute_714, alias_41, permute_716, permute_717, permute_720, div_108, permute_725, permute_729, div_109, permute_734, permute_739, permute_740, alias_42, permute_742, permute_743, permute_746, div_110, permute_751, permute_755, div_111, permute_760, permute_765, permute_766, alias_43, permute_768, permute_769, permute_772, permute_779, div_113, permute_782, permute_786, div_114, permute_791, permute_796, permute_797, alias_44, permute_799, permute_800, permute_803, div_115, permute_808, permute_812, div_116, permute_817, permute_822, permute_823, alias_45, permute_825, permute_826, permute_829, permute_836, div_118, permute_839, permute_843, div_119, permute_848, permute_853, permute_854, alias_46, permute_856, permute_857, permute_860, div_120, permute_865, permute_869, div_121, permute_874, permute_879, permute_880, alias_47, permute_882, permute_883, permute_886, div_122, div_123, tangents_1 = args
    args.clear()
    assert_size_stride(primals_25, (128, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(primals_27, (128, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_56, (256, ), (1, ))
    assert_size_stride(primals_62, (256, ), (1, ))
    assert_size_stride(primals_68, (256, ), (1, ))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_80, (1024, ), (1, ))
    assert_size_stride(primals_83, (512, ), (1, ))
    assert_size_stride(primals_89, (512, ), (1, ))
    assert_size_stride(primals_95, (512, ), (1, ))
    assert_size_stride(primals_101, (512, ), (1, ))
    assert_size_stride(primals_107, (512, ), (1, ))
    assert_size_stride(primals_113, (512, ), (1, ))
    assert_size_stride(primals_119, (512, ), (1, ))
    assert_size_stride(primals_125, (512, ), (1, ))
    assert_size_stride(primals_131, (512, ), (1, ))
    assert_size_stride(primals_137, (512, ), (1, ))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_149, (512, ), (1, ))
    assert_size_stride(primals_155, (512, ), (1, ))
    assert_size_stride(primals_161, (512, ), (1, ))
    assert_size_stride(primals_167, (512, ), (1, ))
    assert_size_stride(primals_173, (512, ), (1, ))
    assert_size_stride(primals_179, (512, ), (1, ))
    assert_size_stride(primals_185, (512, ), (1, ))
    assert_size_stride(primals_191, (512, ), (1, ))
    assert_size_stride(primals_197, (512, ), (1, ))
    assert_size_stride(primals_203, (512, ), (1, ))
    assert_size_stride(primals_209, (512, ), (1, ))
    assert_size_stride(primals_215, (512, ), (1, ))
    assert_size_stride(primals_221, (512, ), (1, ))
    assert_size_stride(primals_227, (512, ), (1, ))
    assert_size_stride(primals_233, (512, ), (1, ))
    assert_size_stride(primals_239, (512, ), (1, ))
    assert_size_stride(primals_245, (512, ), (1, ))
    assert_size_stride(primals_251, (512, ), (1, ))
    assert_size_stride(primals_257, (512, ), (1, ))
    assert_size_stride(primals_263, (512, ), (1, ))
    assert_size_stride(primals_269, (512, ), (1, ))
    assert_size_stride(primals_275, (512, ), (1, ))
    assert_size_stride(primals_281, (512, ), (1, ))
    assert_size_stride(primals_287, (512, ), (1, ))
    assert_size_stride(primals_293, (512, ), (1, ))
    assert_size_stride(primals_299, (2048, ), (1, ))
    assert_size_stride(primals_302, (1024, ), (1, ))
    assert_size_stride(primals_308, (1024, ), (1, ))
    assert_size_stride(primals_314, (1024, ), (1, ))
    assert_size_stride(primals_320, (1024, ), (1, ))
    assert_size_stride(primals_326, (1024, ), (1, ))
    assert_size_stride(primals_365, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(mul, (8, 56, 56, 128), (401408, 7168, 128, 1))
    assert_size_stride(mul_2, (8, 56, 56, 128), (401408, 7168, 128, 1))
    assert_size_stride(view_3, (25088, 128), (128, 1))
    assert_size_stride(view_9, (2401, ), (1, ))
    assert_size_stride(view_15, (25088, 128), (128, 1))
    assert_size_stride(mul_5, (8, 3136, 128), (401408, 128, 1))
    assert_size_stride(view_21, (25088, 128), (128, 1))
    assert_size_stride(addmm_2, (25088, 512), (512, 1))
    assert_size_stride(view_23, (25088, 512), (512, 1))
    assert_size_stride(mul_10, (8, 56, 56, 128), (401408, 7168, 128, 1))
    assert_size_stride(view_29, (25088, 128), (128, 1))
    assert_size_stride(view_35, (2401, ), (1, ))
    assert_size_stride(view_43, (25088, 128), (128, 1))
    assert_size_stride(bernoulli, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_14, (8, 3136, 128), (401408, 128, 1))
    assert_size_stride(view_49, (25088, 128), (128, 1))
    assert_size_stride(addmm_6, (25088, 512), (512, 1))
    assert_size_stride(view_51, (25088, 512), (512, 1))
    assert_size_stride(bernoulli_1, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_20, (8, 28, 28, 512), (401408, 14336, 512, 1))
    assert_size_stride(view_56, (6272, 512), (512, 1))
    assert_size_stride(mm, (6272, 256), (256, 1))
    assert_size_stride(getitem_19, (8, 28, 28, 1), (784, 28, 1, 1))
    assert_size_stride(rsqrt_6, (8, 28, 28, 1), (784, 28, 1, 1))
    assert_size_stride(view_61, (6272, 256), (256, 1))
    assert_size_stride(view_67, (2401, ), (1, ))
    assert_size_stride(view_73, (6272, 256), (256, 1))
    assert_size_stride(bernoulli_2, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_26, (8, 784, 256), (200704, 256, 1))
    assert_size_stride(view_79, (6272, 256), (256, 1))
    assert_size_stride(addmm_10, (6272, 1024), (1024, 1))
    assert_size_stride(view_81, (6272, 1024), (1024, 1))
    assert_size_stride(bernoulli_3, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_32, (8, 28, 28, 256), (200704, 7168, 256, 1))
    assert_size_stride(view_87, (6272, 256), (256, 1))
    assert_size_stride(view_93, (2401, ), (1, ))
    assert_size_stride(view_101, (6272, 256), (256, 1))
    assert_size_stride(bernoulli_4, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_36, (8, 784, 256), (200704, 256, 1))
    assert_size_stride(view_107, (6272, 256), (256, 1))
    assert_size_stride(addmm_14, (6272, 1024), (1024, 1))
    assert_size_stride(view_109, (6272, 1024), (1024, 1))
    assert_size_stride(bernoulli_5, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_42, (8, 14, 14, 1024), (200704, 14336, 1024, 1))
    assert_size_stride(view_114, (1568, 1024), (1024, 1))
    assert_size_stride(mm_1, (1568, 512), (512, 1))
    assert_size_stride(getitem_35, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(rsqrt_11, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(view_119, (1568, 512), (512, 1))
    assert_size_stride(view_125, (2401, ), (1, ))
    assert_size_stride(view_131, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_6, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_48, (8, 196, 512), (100352, 512, 1))
    assert_size_stride(view_137, (1568, 512), (512, 1))
    assert_size_stride(addmm_18, (1568, 2048), (2048, 1))
    assert_size_stride(view_139, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_7, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_54, (8, 14, 14, 512), (100352, 7168, 512, 1))
    assert_size_stride(view_145, (1568, 512), (512, 1))
    assert_size_stride(view_151, (2401, ), (1, ))
    assert_size_stride(view_159, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_8, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_58, (8, 196, 512), (100352, 512, 1))
    assert_size_stride(view_165, (1568, 512), (512, 1))
    assert_size_stride(addmm_22, (1568, 2048), (2048, 1))
    assert_size_stride(view_167, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_9, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_64, (8, 14, 14, 512), (100352, 7168, 512, 1))
    assert_size_stride(view_173, (1568, 512), (512, 1))
    assert_size_stride(view_179, (2401, ), (1, ))
    assert_size_stride(view_185, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_10, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_68, (8, 196, 512), (100352, 512, 1))
    assert_size_stride(view_191, (1568, 512), (512, 1))
    assert_size_stride(addmm_26, (1568, 2048), (2048, 1))
    assert_size_stride(view_193, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_11, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_74, (8, 14, 14, 512), (100352, 7168, 512, 1))
    assert_size_stride(view_199, (1568, 512), (512, 1))
    assert_size_stride(view_205, (2401, ), (1, ))
    assert_size_stride(view_213, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_12, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_78, (8, 196, 512), (100352, 512, 1))
    assert_size_stride(view_219, (1568, 512), (512, 1))
    assert_size_stride(addmm_30, (1568, 2048), (2048, 1))
    assert_size_stride(view_221, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_13, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_84, (8, 14, 14, 512), (100352, 7168, 512, 1))
    assert_size_stride(view_227, (1568, 512), (512, 1))
    assert_size_stride(view_233, (2401, ), (1, ))
    assert_size_stride(view_239, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_14, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_88, (8, 196, 512), (100352, 512, 1))
    assert_size_stride(view_245, (1568, 512), (512, 1))
    assert_size_stride(addmm_34, (1568, 2048), (2048, 1))
    assert_size_stride(view_247, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_15, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_94, (8, 14, 14, 512), (100352, 7168, 512, 1))
    assert_size_stride(view_253, (1568, 512), (512, 1))
    assert_size_stride(view_259, (2401, ), (1, ))
    assert_size_stride(view_267, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_16, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_98, (8, 196, 512), (100352, 512, 1))
    assert_size_stride(view_273, (1568, 512), (512, 1))
    assert_size_stride(addmm_38, (1568, 2048), (2048, 1))
    assert_size_stride(view_275, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_17, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_104, (8, 14, 14, 512), (100352, 7168, 512, 1))
    assert_size_stride(view_281, (1568, 512), (512, 1))
    assert_size_stride(view_287, (2401, ), (1, ))
    assert_size_stride(view_293, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_18, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_108, (8, 196, 512), (100352, 512, 1))
    assert_size_stride(view_299, (1568, 512), (512, 1))
    assert_size_stride(addmm_42, (1568, 2048), (2048, 1))
    assert_size_stride(view_301, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_19, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_114, (8, 14, 14, 512), (100352, 7168, 512, 1))
    assert_size_stride(view_307, (1568, 512), (512, 1))
    assert_size_stride(view_313, (2401, ), (1, ))
    assert_size_stride(view_321, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_20, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_118, (8, 196, 512), (100352, 512, 1))
    assert_size_stride(view_327, (1568, 512), (512, 1))
    assert_size_stride(addmm_46, (1568, 2048), (2048, 1))
    assert_size_stride(view_329, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_21, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_124, (8, 14, 14, 512), (100352, 7168, 512, 1))
    assert_size_stride(view_335, (1568, 512), (512, 1))
    assert_size_stride(view_341, (2401, ), (1, ))
    assert_size_stride(view_347, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_22, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_128, (8, 196, 512), (100352, 512, 1))
    assert_size_stride(view_353, (1568, 512), (512, 1))
    assert_size_stride(addmm_50, (1568, 2048), (2048, 1))
    assert_size_stride(view_355, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_23, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_134, (8, 14, 14, 512), (100352, 7168, 512, 1))
    assert_size_stride(view_361, (1568, 512), (512, 1))
    assert_size_stride(view_367, (2401, ), (1, ))
    assert_size_stride(view_375, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_24, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_138, (8, 196, 512), (100352, 512, 1))
    assert_size_stride(view_381, (1568, 512), (512, 1))
    assert_size_stride(addmm_54, (1568, 2048), (2048, 1))
    assert_size_stride(view_383, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_25, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_144, (8, 14, 14, 512), (100352, 7168, 512, 1))
    assert_size_stride(view_389, (1568, 512), (512, 1))
    assert_size_stride(view_395, (2401, ), (1, ))
    assert_size_stride(view_401, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_26, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_148, (8, 196, 512), (100352, 512, 1))
    assert_size_stride(view_407, (1568, 512), (512, 1))
    assert_size_stride(addmm_58, (1568, 2048), (2048, 1))
    assert_size_stride(view_409, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_27, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_154, (8, 14, 14, 512), (100352, 7168, 512, 1))
    assert_size_stride(view_415, (1568, 512), (512, 1))
    assert_size_stride(view_421, (2401, ), (1, ))
    assert_size_stride(view_429, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_28, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_158, (8, 196, 512), (100352, 512, 1))
    assert_size_stride(view_435, (1568, 512), (512, 1))
    assert_size_stride(addmm_62, (1568, 2048), (2048, 1))
    assert_size_stride(view_437, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_29, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_164, (8, 14, 14, 512), (100352, 7168, 512, 1))
    assert_size_stride(view_443, (1568, 512), (512, 1))
    assert_size_stride(view_449, (2401, ), (1, ))
    assert_size_stride(view_455, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_30, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_168, (8, 196, 512), (100352, 512, 1))
    assert_size_stride(view_461, (1568, 512), (512, 1))
    assert_size_stride(addmm_66, (1568, 2048), (2048, 1))
    assert_size_stride(view_463, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_31, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_174, (8, 14, 14, 512), (100352, 7168, 512, 1))
    assert_size_stride(view_469, (1568, 512), (512, 1))
    assert_size_stride(view_475, (2401, ), (1, ))
    assert_size_stride(view_483, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_32, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_178, (8, 196, 512), (100352, 512, 1))
    assert_size_stride(view_489, (1568, 512), (512, 1))
    assert_size_stride(addmm_70, (1568, 2048), (2048, 1))
    assert_size_stride(view_491, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_33, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_184, (8, 14, 14, 512), (100352, 7168, 512, 1))
    assert_size_stride(view_497, (1568, 512), (512, 1))
    assert_size_stride(view_503, (2401, ), (1, ))
    assert_size_stride(view_509, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_34, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_188, (8, 196, 512), (100352, 512, 1))
    assert_size_stride(view_515, (1568, 512), (512, 1))
    assert_size_stride(addmm_74, (1568, 2048), (2048, 1))
    assert_size_stride(view_517, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_35, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_194, (8, 14, 14, 512), (100352, 7168, 512, 1))
    assert_size_stride(view_523, (1568, 512), (512, 1))
    assert_size_stride(view_529, (2401, ), (1, ))
    assert_size_stride(view_537, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_36, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_198, (8, 196, 512), (100352, 512, 1))
    assert_size_stride(view_543, (1568, 512), (512, 1))
    assert_size_stride(addmm_78, (1568, 2048), (2048, 1))
    assert_size_stride(view_545, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_37, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_204, (8, 14, 14, 512), (100352, 7168, 512, 1))
    assert_size_stride(view_551, (1568, 512), (512, 1))
    assert_size_stride(view_557, (2401, ), (1, ))
    assert_size_stride(view_563, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_38, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_208, (8, 196, 512), (100352, 512, 1))
    assert_size_stride(view_569, (1568, 512), (512, 1))
    assert_size_stride(addmm_82, (1568, 2048), (2048, 1))
    assert_size_stride(view_571, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_39, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_214, (8, 14, 14, 512), (100352, 7168, 512, 1))
    assert_size_stride(view_577, (1568, 512), (512, 1))
    assert_size_stride(view_583, (2401, ), (1, ))
    assert_size_stride(view_591, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_40, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_218, (8, 196, 512), (100352, 512, 1))
    assert_size_stride(view_597, (1568, 512), (512, 1))
    assert_size_stride(addmm_86, (1568, 2048), (2048, 1))
    assert_size_stride(view_599, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_41, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_224, (8, 7, 7, 2048), (100352, 14336, 2048, 1))
    assert_size_stride(view_604, (392, 2048), (2048, 1))
    assert_size_stride(mm_2, (392, 1024), (1024, 1))
    assert_size_stride(getitem_163, (8, 7, 7, 1), (49, 7, 1, 1))
    assert_size_stride(rsqrt_48, (8, 7, 7, 1), (49, 7, 1, 1))
    assert_size_stride(view_609, (392, 1024), (1024, 1))
    assert_size_stride(view_615, (2401, ), (1, ))
    assert_size_stride(view_621, (392, 1024), (1024, 1))
    assert_size_stride(bernoulli_42, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_230, (8, 49, 1024), (50176, 1024, 1))
    assert_size_stride(view_627, (392, 1024), (1024, 1))
    assert_size_stride(addmm_90, (392, 4096), (4096, 1))
    assert_size_stride(view_629, (392, 4096), (4096, 1))
    assert_size_stride(bernoulli_43, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_236, (8, 7, 7, 1024), (50176, 7168, 1024, 1))
    assert_size_stride(view_635, (392, 1024), (1024, 1))
    assert_size_stride(view_641, (2401, ), (1, ))
    assert_size_stride(view_647, (392, 1024), (1024, 1))
    assert_size_stride(bernoulli_44, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_240, (8, 49, 1024), (50176, 1024, 1))
    assert_size_stride(view_653, (392, 1024), (1024, 1))
    assert_size_stride(addmm_94, (392, 4096), (4096, 1))
    assert_size_stride(view_655, (392, 4096), (4096, 1))
    assert_size_stride(bernoulli_45, (8, 1, 1), (1, 1, 1))
    assert_size_stride(mul_246, (8, 7, 7, 1024), (50176, 7168, 1024, 1))
    assert_size_stride(clone_264, (8, 1024), (1024, 1))
    assert_size_stride(permute_248, (1000, 1024), (1024, 1))
    assert_size_stride(div_71, (8, 7, 7, 1), (49, 7, 1, 1))
    assert_size_stride(permute_252, (1024, 4096), (4096, 1))
    assert_size_stride(permute_256, (4096, 1024), (1024, 1))
    assert_size_stride(div_72, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_261, (1024, 1024), (1024, 1))
    assert_size_stride(permute_266, (256, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_267, (256, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_24, (8, 32, 49, 49), (76832, 2401, 49, 1))
    assert_size_stride(permute_269, (256, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_270, (256, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_273, (3072, 1024), (1024, 1))
    assert_size_stride(div_73, (8, 7, 7, 1), (49, 7, 1, 1))
    assert_size_stride(permute_278, (1024, 4096), (4096, 1))
    assert_size_stride(permute_282, (4096, 1024), (1024, 1))
    assert_size_stride(div_74, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_287, (1024, 1024), (1024, 1))
    assert_size_stride(permute_292, (256, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_293, (256, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_25, (8, 32, 49, 49), (76832, 2401, 49, 1))
    assert_size_stride(permute_295, (256, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_296, (256, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_299, (3072, 1024), (1024, 1))
    assert_size_stride(permute_306, (1024, 2048), (2048, 1))
    assert_size_stride(div_76, (8, 7, 7, 1), (49, 7, 1, 1))
    assert_size_stride(permute_309, (512, 2048), (2048, 1))
    assert_size_stride(permute_313, (2048, 512), (512, 1))
    assert_size_stride(div_77, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_318, (512, 512), (512, 1))
    assert_size_stride(permute_323, (512, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_324, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_26, (32, 16, 49, 49), (38416, 2401, 49, 1))
    assert_size_stride(permute_326, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_327, (512, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_330, (1536, 512), (512, 1))
    assert_size_stride(div_78, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_335, (512, 2048), (2048, 1))
    assert_size_stride(permute_339, (2048, 512), (512, 1))
    assert_size_stride(div_79, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_344, (512, 512), (512, 1))
    assert_size_stride(permute_349, (512, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_350, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_27, (32, 16, 49, 49), (38416, 2401, 49, 1))
    assert_size_stride(permute_352, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_353, (512, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_356, (1536, 512), (512, 1))
    assert_size_stride(div_80, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_361, (512, 2048), (2048, 1))
    assert_size_stride(permute_365, (2048, 512), (512, 1))
    assert_size_stride(div_81, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_370, (512, 512), (512, 1))
    assert_size_stride(permute_375, (512, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_376, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_28, (32, 16, 49, 49), (38416, 2401, 49, 1))
    assert_size_stride(permute_378, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_379, (512, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_382, (1536, 512), (512, 1))
    assert_size_stride(div_82, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_387, (512, 2048), (2048, 1))
    assert_size_stride(permute_391, (2048, 512), (512, 1))
    assert_size_stride(div_83, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_396, (512, 512), (512, 1))
    assert_size_stride(permute_401, (512, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_402, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_29, (32, 16, 49, 49), (38416, 2401, 49, 1))
    assert_size_stride(permute_404, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_405, (512, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_408, (1536, 512), (512, 1))
    assert_size_stride(div_84, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_413, (512, 2048), (2048, 1))
    assert_size_stride(permute_417, (2048, 512), (512, 1))
    assert_size_stride(div_85, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_422, (512, 512), (512, 1))
    assert_size_stride(permute_427, (512, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_428, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_30, (32, 16, 49, 49), (38416, 2401, 49, 1))
    assert_size_stride(permute_430, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_431, (512, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_434, (1536, 512), (512, 1))
    assert_size_stride(div_86, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_439, (512, 2048), (2048, 1))
    assert_size_stride(permute_443, (2048, 512), (512, 1))
    assert_size_stride(div_87, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_448, (512, 512), (512, 1))
    assert_size_stride(permute_453, (512, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_454, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_31, (32, 16, 49, 49), (38416, 2401, 49, 1))
    assert_size_stride(permute_456, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_457, (512, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_460, (1536, 512), (512, 1))
    assert_size_stride(div_88, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_465, (512, 2048), (2048, 1))
    assert_size_stride(permute_469, (2048, 512), (512, 1))
    assert_size_stride(div_89, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_474, (512, 512), (512, 1))
    assert_size_stride(permute_479, (512, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_480, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_32, (32, 16, 49, 49), (38416, 2401, 49, 1))
    assert_size_stride(permute_482, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_483, (512, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_486, (1536, 512), (512, 1))
    assert_size_stride(div_90, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_491, (512, 2048), (2048, 1))
    assert_size_stride(permute_495, (2048, 512), (512, 1))
    assert_size_stride(div_91, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_500, (512, 512), (512, 1))
    assert_size_stride(permute_505, (512, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_506, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_33, (32, 16, 49, 49), (38416, 2401, 49, 1))
    assert_size_stride(permute_508, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_509, (512, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_512, (1536, 512), (512, 1))
    assert_size_stride(div_92, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_517, (512, 2048), (2048, 1))
    assert_size_stride(permute_521, (2048, 512), (512, 1))
    assert_size_stride(div_93, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_526, (512, 512), (512, 1))
    assert_size_stride(permute_531, (512, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_532, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_34, (32, 16, 49, 49), (38416, 2401, 49, 1))
    assert_size_stride(permute_534, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_535, (512, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_538, (1536, 512), (512, 1))
    assert_size_stride(div_94, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_543, (512, 2048), (2048, 1))
    assert_size_stride(permute_547, (2048, 512), (512, 1))
    assert_size_stride(div_95, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_552, (512, 512), (512, 1))
    assert_size_stride(permute_557, (512, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_558, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_35, (32, 16, 49, 49), (38416, 2401, 49, 1))
    assert_size_stride(permute_560, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_561, (512, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_564, (1536, 512), (512, 1))
    assert_size_stride(div_96, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_569, (512, 2048), (2048, 1))
    assert_size_stride(permute_573, (2048, 512), (512, 1))
    assert_size_stride(div_97, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_578, (512, 512), (512, 1))
    assert_size_stride(permute_583, (512, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_584, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_36, (32, 16, 49, 49), (38416, 2401, 49, 1))
    assert_size_stride(permute_586, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_587, (512, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_590, (1536, 512), (512, 1))
    assert_size_stride(div_98, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_595, (512, 2048), (2048, 1))
    assert_size_stride(permute_599, (2048, 512), (512, 1))
    assert_size_stride(div_99, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_604, (512, 512), (512, 1))
    assert_size_stride(permute_609, (512, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_610, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_37, (32, 16, 49, 49), (38416, 2401, 49, 1))
    assert_size_stride(permute_612, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_613, (512, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_616, (1536, 512), (512, 1))
    assert_size_stride(div_100, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_621, (512, 2048), (2048, 1))
    assert_size_stride(permute_625, (2048, 512), (512, 1))
    assert_size_stride(div_101, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_630, (512, 512), (512, 1))
    assert_size_stride(permute_635, (512, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_636, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_38, (32, 16, 49, 49), (38416, 2401, 49, 1))
    assert_size_stride(permute_638, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_639, (512, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_642, (1536, 512), (512, 1))
    assert_size_stride(div_102, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_647, (512, 2048), (2048, 1))
    assert_size_stride(permute_651, (2048, 512), (512, 1))
    assert_size_stride(div_103, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_656, (512, 512), (512, 1))
    assert_size_stride(permute_661, (512, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_662, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_39, (32, 16, 49, 49), (38416, 2401, 49, 1))
    assert_size_stride(permute_664, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_665, (512, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_668, (1536, 512), (512, 1))
    assert_size_stride(div_104, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_673, (512, 2048), (2048, 1))
    assert_size_stride(permute_677, (2048, 512), (512, 1))
    assert_size_stride(div_105, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_682, (512, 512), (512, 1))
    assert_size_stride(permute_687, (512, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_688, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_40, (32, 16, 49, 49), (38416, 2401, 49, 1))
    assert_size_stride(permute_690, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_691, (512, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_694, (1536, 512), (512, 1))
    assert_size_stride(div_106, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_699, (512, 2048), (2048, 1))
    assert_size_stride(permute_703, (2048, 512), (512, 1))
    assert_size_stride(div_107, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_708, (512, 512), (512, 1))
    assert_size_stride(permute_713, (512, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_714, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_41, (32, 16, 49, 49), (38416, 2401, 49, 1))
    assert_size_stride(permute_716, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_717, (512, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_720, (1536, 512), (512, 1))
    assert_size_stride(div_108, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_725, (512, 2048), (2048, 1))
    assert_size_stride(permute_729, (2048, 512), (512, 1))
    assert_size_stride(div_109, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_734, (512, 512), (512, 1))
    assert_size_stride(permute_739, (512, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_740, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_42, (32, 16, 49, 49), (38416, 2401, 49, 1))
    assert_size_stride(permute_742, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_743, (512, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_746, (1536, 512), (512, 1))
    assert_size_stride(div_110, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_751, (512, 2048), (2048, 1))
    assert_size_stride(permute_755, (2048, 512), (512, 1))
    assert_size_stride(div_111, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_760, (512, 512), (512, 1))
    assert_size_stride(permute_765, (512, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_766, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_43, (32, 16, 49, 49), (38416, 2401, 49, 1))
    assert_size_stride(permute_768, (512, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_769, (512, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_772, (1536, 512), (512, 1))
    assert_size_stride(permute_779, (512, 1024), (1024, 1))
    assert_size_stride(div_113, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_782, (256, 1024), (1024, 1))
    assert_size_stride(permute_786, (1024, 256), (256, 1))
    assert_size_stride(div_114, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_791, (256, 256), (256, 1))
    assert_size_stride(permute_796, (1024, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_797, (1024, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_44, (128, 8, 49, 49), (19208, 2401, 49, 1))
    assert_size_stride(permute_799, (1024, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_800, (1024, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_803, (768, 256), (256, 1))
    assert_size_stride(div_115, (8, 28, 28, 1), (784, 28, 1, 1))
    assert_size_stride(permute_808, (256, 1024), (1024, 1))
    assert_size_stride(permute_812, (1024, 256), (256, 1))
    assert_size_stride(div_116, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_817, (256, 256), (256, 1))
    assert_size_stride(permute_822, (1024, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_823, (1024, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_45, (128, 8, 49, 49), (19208, 2401, 49, 1))
    assert_size_stride(permute_825, (1024, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_826, (1024, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_829, (768, 256), (256, 1))
    assert_size_stride(permute_836, (256, 512), (512, 1))
    assert_size_stride(div_118, (8, 28, 28, 1), (784, 28, 1, 1))
    assert_size_stride(permute_839, (128, 512), (512, 1))
    assert_size_stride(permute_843, (512, 128), (128, 1))
    assert_size_stride(div_119, (8, 3136, 1), (3136, 1, 1))
    assert_size_stride(permute_848, (128, 128), (128, 1))
    assert_size_stride(permute_853, (2048, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_854, (2048, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_46, (512, 4, 49, 49), (9604, 2401, 49, 1))
    assert_size_stride(permute_856, (2048, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_857, (2048, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_860, (384, 128), (128, 1))
    assert_size_stride(div_120, (8, 56, 56, 1), (3136, 56, 1, 1))
    assert_size_stride(permute_865, (128, 512), (512, 1))
    assert_size_stride(permute_869, (512, 128), (128, 1))
    assert_size_stride(div_121, (8, 3136, 1), (3136, 1, 1))
    assert_size_stride(permute_874, (128, 128), (128, 1))
    assert_size_stride(permute_879, (2048, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_880, (2048, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_47, (512, 4, 49, 49), (9604, 2401, 49, 1))
    assert_size_stride(permute_882, (2048, 32, 49), (1568, 1, 32))
    assert_size_stride(permute_883, (2048, 49, 32), (1568, 1, 49))
    assert_size_stride(permute_886, (384, 128), (128, 1))
    assert_size_stride(div_122, (8, 56, 56, 1), (3136, 56, 1, 1))
    assert_size_stride(div_123, (8, 56, 56, 1), (3136, 56, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_248, out=buf0)
        del permute_248
        buf1 = empty((1000, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_264, out=buf1)
        del clone_264
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf8 = empty((8, 7, 7, 1024), device='cuda', dtype=torch.float32)
        buf9 = empty((8, 49, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__45], Original ATen: [aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_red_fused_div_mul_native_layer_norm_backward_1.run(buf0, primals_326, mul_246, div_71, bernoulli_45, buf8, buf9, 392, 1024, grid=grid(392), stream=stream0)
        del bernoulli_45
        del div_71
        del primals_326
        buf5 = empty_strided((1024, 4), (1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]
        triton_red_fused_div_native_layer_norm_backward_2.run(buf0, mul_246, buf5, 4096, 98, grid=grid(4096), stream=stream0)
        del mul_246
        buf6 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf5, buf6, 1024, 4, grid=grid(1024), stream=stream0)
        buf7 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]
        triton_red_fused_div_native_layer_norm_backward_4.run(buf0, buf7, 1024, 392, grid=grid(1024), stream=stream0)
        buf10 = empty((392, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf9, (392, 1024), (1024, 1), 0), permute_252, out=buf10)
        del permute_252
        buf11 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf9, (1024, 392), (1, 1024), 0), view_655, out=buf11)
        del view_655
        buf12 = reinterpret_tensor(buf5, (1, 1024, 4), (4096, 1, 1024), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf9, buf12, 4096, 98, grid=grid(4096), stream=stream0)
        buf13 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf12, buf13, 1024, 4, grid=grid(1024), stream=stream0)
        buf14 = reinterpret_tensor(buf10, (8, 49, 4096), (200704, 4096, 1), 0); del buf10  # reuse
        # Source Nodes: [x_446], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf14, addmm_94, 1605632, grid=grid(1605632), stream=stream0)
        del addmm_94
        buf15 = reinterpret_tensor(buf9, (392, 1024), (1024, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (392, 4096), (4096, 1), 0), permute_256, out=buf15)
        del permute_256
        buf16 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (4096, 392), (1, 4096), 0), view_653, out=buf16)
        del view_653
        buf17 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf14, buf17, 16384, 98, grid=grid(16384), stream=stream0)
        buf18 = reinterpret_tensor(buf12, (1, 4096), (4096, 1), 0); del buf12  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf17, buf18, 4096, 4, grid=grid(4096), stream=stream0)
        buf25 = reinterpret_tensor(buf8, (8, 49, 1024), (50176, 1024, 1), 0); del buf8  # reuse
        buf26 = empty((8, 7, 7, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__44], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_9.run(buf25, buf15, primals_320, mul_240, div_72, bernoulli_44, buf26, 392, 1024, grid=grid(392), stream=stream0)
        del bernoulli_44
        del div_72
        del primals_320
        buf21 = empty_strided((1024, 4), (1, 1024), device='cuda', dtype=torch.float32)
        buf23 = empty_strided((1024, 4), (1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf15, mul_240, buf21, buf23, 4096, 98, grid=grid(4096), stream=stream0)
        del mul_240
        buf22 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf21, buf22, 1024, 4, grid=grid(1024), stream=stream0)
        buf24 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf23, buf24, 1024, 4, grid=grid(1024), stream=stream0)
        buf27 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (392, 1024), (1024, 1), 0), permute_261, out=buf27)
        del permute_261
        buf28 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (1024, 392), (1, 1024), 0), view_647, out=buf28)
        del view_647
        buf29 = reinterpret_tensor(buf23, (1, 1024, 4), (4096, 1, 1024), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf26, buf29, 4096, 98, grid=grid(4096), stream=stream0)
        buf30 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf29, buf30, 1024, 4, grid=grid(1024), stream=stream0)
        buf31 = reinterpret_tensor(buf26, (8, 32, 49, 32), (50176, 1568, 32, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf27, buf31, 401408, grid=grid(401408), stream=stream0)
        buf32 = reinterpret_tensor(buf27, (256, 49, 32), (1568, 32, 1), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_266, reinterpret_tensor(buf31, (256, 49, 32), (1568, 32, 1), 0), out=buf32)
        del permute_266
        buf33 = empty((256, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf31, (256, 49, 32), (1568, 32, 1), 0), permute_267, out=buf33)
        del permute_267
        buf34 = empty_strided((8, 32, 49, 1), (1568, 49, 1, 12544), device='cuda', dtype=torch.float32)
        buf39 = empty((8, 32, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_12.run(buf33, alias_24, buf34, buf39, 12544, 49, grid=grid(12544), stream=stream0)
        buf35 = empty((1, 32, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused__softmax_backward_data_sum_13.run(buf33, alias_24, buf34, buf35, 76832, 8, grid=grid(76832), stream=stream0)
        del alias_24
        buf36 = empty((169, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_14.run(buf36, 5408, grid=grid(5408), stream=stream0)
        aten.index_put_(buf36, [view_641], reinterpret_tensor(buf35, (2401, 32), (1, 2401), 0), True)
        del view_641
        buf40 = reinterpret_tensor(buf31, (256, 32, 49), (1568, 49, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_269, reinterpret_tensor(buf39, (256, 49, 49), (2401, 49, 1), 0), out=buf40)
        del permute_269
        buf41 = empty((256, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf39, (256, 49, 49), (2401, 49, 1), 0), permute_270, out=buf41)
        del permute_270
        buf42 = empty((8, 49, 3, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf41, buf40, buf32, buf42, 37632, 32, grid=grid(37632, 32), stream=stream0)
        buf43 = reinterpret_tensor(buf41, (392, 1024), (1024, 1), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf42, (392, 3072), (3072, 1), 0), permute_273, out=buf43)
        del permute_273
        buf44 = empty((3072, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf42, (3072, 392), (1, 3072), 0), view_635, out=buf44)
        del view_635
        buf45 = empty_strided((1, 3072, 4), (12288, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf42, buf45, 12288, 98, grid=grid(12288), stream=stream0)
        buf46 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf45, buf46, 3072, 4, grid=grid(3072), stream=stream0)
        buf53 = reinterpret_tensor(buf25, (8, 7, 7, 1024), (50176, 7168, 1024, 1), 0); del buf25  # reuse
        buf54 = reinterpret_tensor(buf40, (8, 49, 1024), (50176, 1024, 1), 0); del buf40  # reuse
        # Source Nodes: [div__43], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_18.run(buf53, buf43, primals_314, mul_236, div_73, bernoulli_43, buf54, 392, 1024, grid=grid(392), stream=stream0)
        del bernoulli_43
        del div_73
        del primals_314
        buf49 = reinterpret_tensor(buf29, (1024, 4), (1, 1024), 0); del buf29  # reuse
        buf51 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf43, mul_236, buf49, buf51, 4096, 98, grid=grid(4096), stream=stream0)
        del mul_236
        buf50 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf49, buf50, 1024, 4, grid=grid(1024), stream=stream0)
        buf52 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf51, buf52, 1024, 4, grid=grid(1024), stream=stream0)
        buf55 = reinterpret_tensor(buf14, (392, 4096), (4096, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (392, 1024), (1024, 1), 0), permute_278, out=buf55)
        del permute_278
        buf56 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (1024, 392), (1, 1024), 0), view_629, out=buf56)
        del view_629
        buf57 = reinterpret_tensor(buf51, (1, 1024, 4), (4096, 1, 1024), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf54, buf57, 4096, 98, grid=grid(4096), stream=stream0)
        buf58 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf57, buf58, 1024, 4, grid=grid(1024), stream=stream0)
        buf59 = reinterpret_tensor(buf55, (8, 49, 4096), (200704, 4096, 1), 0); del buf55  # reuse
        # Source Nodes: [x_428], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf59, addmm_90, 1605632, grid=grid(1605632), stream=stream0)
        del addmm_90
        buf60 = reinterpret_tensor(buf54, (392, 1024), (1024, 1), 0); del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (392, 4096), (4096, 1), 0), permute_282, out=buf60)
        del permute_282
        buf61 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (4096, 392), (1, 4096), 0), view_627, out=buf61)
        del view_627
        buf62 = buf17; del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf59, buf62, 16384, 98, grid=grid(16384), stream=stream0)
        buf63 = reinterpret_tensor(buf57, (1, 4096), (4096, 1), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf62, buf63, 4096, 4, grid=grid(4096), stream=stream0)
        buf70 = reinterpret_tensor(buf53, (8, 49, 1024), (50176, 1024, 1), 0); del buf53  # reuse
        buf71 = reinterpret_tensor(buf43, (8, 7, 7, 1024), (50176, 7168, 1024, 1), 0); del buf43  # reuse
        # Source Nodes: [div__42], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_18.run(buf70, buf60, primals_308, mul_230, div_74, bernoulli_42, buf71, 392, 1024, grid=grid(392), stream=stream0)
        del bernoulli_42
        del div_74
        del primals_308
        buf66 = buf49; del buf49  # reuse
        buf68 = empty_strided((1024, 4), (1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf60, mul_230, buf66, buf68, 4096, 98, grid=grid(4096), stream=stream0)
        del mul_230
        buf67 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf66, buf67, 1024, 4, grid=grid(1024), stream=stream0)
        buf69 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf68, buf69, 1024, 4, grid=grid(1024), stream=stream0)
        buf72 = buf60; del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf71, (392, 1024), (1024, 1), 0), permute_287, out=buf72)
        del permute_287
        buf73 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf71, (1024, 392), (1, 1024), 0), view_621, out=buf73)
        del view_621
        buf74 = reinterpret_tensor(buf68, (1, 1024, 4), (4096, 1, 1024), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf71, buf74, 4096, 98, grid=grid(4096), stream=stream0)
        buf75 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf74, buf75, 1024, 4, grid=grid(1024), stream=stream0)
        buf76 = reinterpret_tensor(buf71, (8, 32, 49, 32), (50176, 1568, 32, 1), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf72, buf76, 401408, grid=grid(401408), stream=stream0)
        buf77 = reinterpret_tensor(buf72, (256, 49, 32), (1568, 32, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_292, reinterpret_tensor(buf76, (256, 49, 32), (1568, 32, 1), 0), out=buf77)
        del permute_292
        buf78 = reinterpret_tensor(buf39, (256, 49, 49), (2401, 49, 1), 0); del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf76, (256, 49, 32), (1568, 32, 1), 0), permute_293, out=buf78)
        del permute_293
        buf79 = buf34; del buf34  # reuse
        buf84 = reinterpret_tensor(buf33, (8, 32, 49, 49), (76832, 2401, 49, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_12.run(buf78, alias_25, buf79, buf84, 12544, 49, grid=grid(12544), stream=stream0)
        buf80 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused__softmax_backward_data_sum_13.run(buf78, alias_25, buf79, buf80, 76832, 8, grid=grid(76832), stream=stream0)
        del alias_25
        del buf78
        buf81 = empty((169, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_14.run(buf81, 5408, grid=grid(5408), stream=stream0)
        aten.index_put_(buf81, [view_615], reinterpret_tensor(buf80, (2401, 32), (1, 2401), 0), True)
        del buf80
        del view_615
        buf85 = reinterpret_tensor(buf76, (256, 32, 49), (1568, 49, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_295, reinterpret_tensor(buf84, (256, 49, 49), (2401, 49, 1), 0), out=buf85)
        del permute_295
        buf86 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf84, (256, 49, 49), (2401, 49, 1), 0), permute_296, out=buf86)
        del buf84
        del permute_296
        buf87 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf86, buf85, buf77, buf87, 37632, 32, grid=grid(37632, 32), stream=stream0)
        del buf77
        del buf85
        buf88 = reinterpret_tensor(buf86, (392, 1024), (1024, 1), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (392, 3072), (3072, 1), 0), permute_299, out=buf88)
        del permute_299
        buf89 = empty((3072, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (3072, 392), (1, 3072), 0), view_609, out=buf89)
        del view_609
        buf90 = buf45; del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf87, buf90, 12288, 98, grid=grid(12288), stream=stream0)
        del buf87
        buf91 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf90, buf91, 3072, 4, grid=grid(3072), stream=stream0)
        del buf90
        buf98 = reinterpret_tensor(buf70, (8, 7, 7, 1024), (50176, 7168, 1024, 1), 0); del buf70  # reuse
        # Source Nodes: [shifted_x_88], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_19.run(buf98, buf88, primals_302, mm_2, getitem_163, rsqrt_48, 392, 1024, grid=grid(392), stream=stream0)
        del primals_302
        buf94 = reinterpret_tensor(buf74, (1024, 4), (1, 1024), 0); del buf74  # reuse
        buf96 = buf66; del buf66  # reuse
        # Source Nodes: [shifted_x_88], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_20.run(buf88, mm_2, getitem_163, rsqrt_48, buf94, buf96, 4096, 98, grid=grid(4096), stream=stream0)
        del buf88
        del getitem_163
        del mm_2
        del rsqrt_48
        buf95 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shifted_x_88], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf94, buf95, 1024, 4, grid=grid(1024), stream=stream0)
        del buf94
        buf97 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf96, buf97, 1024, 4, grid=grid(1024), stream=stream0)
        del buf96
        buf99 = empty((1024, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf98, (1024, 392), (1, 1024), 0), view_604, out=buf99)
        del view_604
        buf100 = empty((392, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf98, (392, 1024), (1024, 1), 0), permute_306, out=buf100)
        del buf98
        del permute_306
        buf107 = empty_strided((8, 7, 2, 7, 2, 512), (100352, 14336, 512, 2048, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone, aten.native_layer_norm_backward]
        triton_red_fused_clone_native_layer_norm_backward_21.run(buf100, primals_299, mul_224, div_76, buf107, 392, 2048, grid=grid(392), stream=stream0)
        del div_76
        del primals_299
        buf103 = reinterpret_tensor(buf0, (2048, 4), (1, 2048), 0); del buf0  # reuse
        buf105 = empty_strided((2048, 4), (1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_22.run(buf100, mul_224, buf103, buf105, 8192, 98, grid=grid(8192), stream=stream0)
        del mul_224
        buf104 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_23.run(buf103, buf104, 2048, 4, grid=grid(2048), stream=stream0)
        del buf103
        buf106 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_23.run(buf105, buf106, 2048, 4, grid=grid(2048), stream=stream0)
        del buf105
        buf108 = reinterpret_tensor(buf100, (8, 196, 512), (100352, 512, 1), 0); del buf100  # reuse
        # Source Nodes: [div__41], Original ATen: [aten.div, aten.mul]
        triton_poi_fused_div_mul_24.run(buf107, bernoulli_41, buf108, 802816, grid=grid(802816), stream=stream0)
        del bernoulli_41
        buf109 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (1568, 512), (512, 1), 0), permute_309, out=buf109)
        del permute_309
        buf110 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (512, 1568), (1, 512), 0), view_599, out=buf110)
        del view_599
        buf111 = empty_strided((1, 512, 13), (6656, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf108, buf111, 6656, 121, grid=grid(6656), stream=stream0)
        buf112 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf111, buf112, 512, 13, grid=grid(512), stream=stream0)
        buf113 = reinterpret_tensor(buf109, (8, 196, 2048), (401408, 2048, 1), 0); del buf109  # reuse
        # Source Nodes: [x_405], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_27.run(buf113, addmm_86, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_86
        buf114 = reinterpret_tensor(buf108, (1568, 512), (512, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf113, (1568, 2048), (2048, 1), 0), permute_313, out=buf114)
        del permute_313
        buf115 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf113, (2048, 1568), (1, 2048), 0), view_597, out=buf115)
        del view_597
        buf116 = empty_strided((1, 2048, 13), (26624, 1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf113, buf116, 26624, 121, grid=grid(26624), stream=stream0)
        buf117 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf116, buf117, 2048, 13, grid=grid(2048), stream=stream0)
        buf124 = empty((8, 196, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_30.run(buf114, primals_293, mul_218, buf107, div_77, buf124, 1568, 512, grid=grid(1568), stream=stream0)
        del div_77
        del primals_293
        buf120 = reinterpret_tensor(buf111, (512, 13), (1, 512), 0); del buf111  # reuse
        buf122 = empty_strided((512, 13), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_31.run(buf114, mul_218, buf120, buf122, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_218
        buf121 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf120, buf121, 512, 13, grid=grid(512), stream=stream0)
        buf123 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf122, buf123, 512, 13, grid=grid(512), stream=stream0)
        buf125 = reinterpret_tensor(buf114, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf124, bernoulli_40, buf125, 802816, grid=grid(802816), stream=stream0)
        del bernoulli_40
        buf126 = reinterpret_tensor(buf107, (1568, 512), (512, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (1568, 512), (512, 1), 0), permute_318, out=buf126)
        del permute_318
        buf127 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (512, 1568), (1, 512), 0), view_591, out=buf127)
        del view_591
        buf128 = reinterpret_tensor(buf122, (1, 512, 13), (6656, 1, 512), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf125, buf128, 6656, 121, grid=grid(6656), stream=stream0)
        buf129 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf128, buf129, 512, 13, grid=grid(512), stream=stream0)
        buf130 = reinterpret_tensor(buf125, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf126, buf130, 802816, grid=grid(802816), stream=stream0)
        buf131 = reinterpret_tensor(buf126, (512, 49, 32), (1568, 32, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_323, reinterpret_tensor(buf130, (512, 49, 32), (1568, 32, 1), 0), out=buf131)
        del permute_323
        buf132 = empty((512, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf130, (512, 49, 32), (1568, 32, 1), 0), permute_324, out=buf132)
        del permute_324
        buf133 = empty_strided((32, 16, 49, 1), (784, 49, 1, 25088), device='cuda', dtype=torch.float32)
        buf138 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_34.run(buf132, alias_26, buf133, buf138, 25088, 49, grid=grid(25088), stream=stream0)
        buf134 = empty((1, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_35.run(buf132, alias_26, buf133, buf134, 38416, 32, grid=grid(38416), stream=stream0)
        del alias_26
        buf135 = empty((169, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_36.run(buf135, 2704, grid=grid(2704), stream=stream0)
        aten.index_put_(buf135, [view_583], reinterpret_tensor(buf134, (2401, 16), (1, 2401), 0), True)
        del view_583
        buf139 = reinterpret_tensor(buf130, (512, 32, 49), (1568, 49, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_326, reinterpret_tensor(buf138, (512, 49, 49), (2401, 49, 1), 0), out=buf139)
        del permute_326
        buf140 = empty((512, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf138, (512, 49, 49), (2401, 49, 1), 0), permute_327, out=buf140)
        del permute_327
        buf141 = empty((32, 49, 3, 16, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf140, buf139, buf131, buf141, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf142 = reinterpret_tensor(buf140, (1568, 512), (512, 1), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (1568, 1536), (1536, 1), 0), permute_330, out=buf142)
        del permute_330
        buf143 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (1536, 1568), (1, 1536), 0), view_577, out=buf143)
        del view_577
        buf144 = empty_strided((1, 1536, 13), (19968, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf141, buf144, 19968, 121, grid=grid(19968), stream=stream0)
        buf145 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_39.run(buf144, buf145, 1536, 13, grid=grid(1536), stream=stream0)
        buf152 = reinterpret_tensor(buf124, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf124  # reuse
        buf153 = reinterpret_tensor(buf139, (8, 196, 512), (100352, 512, 1), 0); del buf139  # reuse
        # Source Nodes: [div__39], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_add_div_mul_native_layer_norm_backward_roll_40.run(buf152, buf142, primals_287, mul_214, div_78, bernoulli_39, buf153, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_39
        del div_78
        del primals_287
        buf148 = reinterpret_tensor(buf128, (512, 13), (13, 1), 0); del buf128  # reuse
        buf150 = reinterpret_tensor(buf120, (512, 13), (13, 1), 0); del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_red_fused_native_layer_norm_backward_roll_41.run(buf142, mul_214, buf148, buf150, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_214
        buf149 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_42.run(buf148, buf149, 512, 13, grid=grid(512), stream=stream0)
        buf151 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_42.run(buf150, buf151, 512, 13, grid=grid(512), stream=stream0)
        buf154 = reinterpret_tensor(buf113, (1568, 2048), (2048, 1), 0); del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf153, (1568, 512), (512, 1), 0), permute_335, out=buf154)
        del permute_335
        buf155 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf153, (512, 1568), (1, 512), 0), view_571, out=buf155)
        del view_571
        buf156 = reinterpret_tensor(buf150, (1, 512, 13), (6656, 1, 512), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf153, buf156, 6656, 121, grid=grid(6656), stream=stream0)
        buf157 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf156, buf157, 512, 13, grid=grid(512), stream=stream0)
        buf158 = reinterpret_tensor(buf154, (8, 196, 2048), (401408, 2048, 1), 0); del buf154  # reuse
        # Source Nodes: [x_387], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_27.run(buf158, addmm_82, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_82
        buf159 = reinterpret_tensor(buf153, (1568, 512), (512, 1), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf158, (1568, 2048), (2048, 1), 0), permute_339, out=buf159)
        del permute_339
        buf160 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf158, (2048, 1568), (1, 2048), 0), view_569, out=buf160)
        del view_569
        buf161 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf158, buf161, 26624, 121, grid=grid(26624), stream=stream0)
        buf162 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf161, buf162, 2048, 13, grid=grid(2048), stream=stream0)
        buf169 = reinterpret_tensor(buf152, (8, 196, 512), (100352, 512, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf169, buf159, primals_281, mul_208, div_79, 1568, 512, grid=grid(1568), stream=stream0)
        del div_79
        del primals_281
        buf165 = reinterpret_tensor(buf156, (512, 13), (1, 512), 0); del buf156  # reuse
        buf167 = reinterpret_tensor(buf148, (512, 13), (1, 512), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_31.run(buf159, mul_208, buf165, buf167, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_208
        buf166 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf165, buf166, 512, 13, grid=grid(512), stream=stream0)
        buf168 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf167, buf168, 512, 13, grid=grid(512), stream=stream0)
        buf170 = reinterpret_tensor(buf159, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf169, bernoulli_38, buf170, 802816, grid=grid(802816), stream=stream0)
        del bernoulli_38
        buf171 = buf142; del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (1568, 512), (512, 1), 0), permute_344, out=buf171)
        del permute_344
        buf172 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (512, 1568), (1, 512), 0), view_563, out=buf172)
        del view_563
        buf173 = reinterpret_tensor(buf167, (1, 512, 13), (6656, 1, 512), 0); del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf170, buf173, 6656, 121, grid=grid(6656), stream=stream0)
        buf174 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf173, buf174, 512, 13, grid=grid(512), stream=stream0)
        buf175 = reinterpret_tensor(buf170, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf171, buf175, 802816, grid=grid(802816), stream=stream0)
        buf176 = reinterpret_tensor(buf171, (512, 49, 32), (1568, 32, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_349, reinterpret_tensor(buf175, (512, 49, 32), (1568, 32, 1), 0), out=buf176)
        del permute_349
        buf177 = reinterpret_tensor(buf138, (512, 49, 49), (2401, 49, 1), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf175, (512, 49, 32), (1568, 32, 1), 0), permute_350, out=buf177)
        del permute_350
        buf178 = buf133; del buf133  # reuse
        buf183 = reinterpret_tensor(buf132, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_34.run(buf177, alias_27, buf178, buf183, 25088, 49, grid=grid(25088), stream=stream0)
        buf179 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused_sum_35.run(buf177, alias_27, buf178, buf179, 38416, 32, grid=grid(38416), stream=stream0)
        del alias_27
        buf180 = empty((169, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_36.run(buf180, 2704, grid=grid(2704), stream=stream0)
        aten.index_put_(buf180, [view_557], reinterpret_tensor(buf179, (2401, 16), (1, 2401), 0), True)
        del view_557
        buf184 = reinterpret_tensor(buf175, (512, 32, 49), (1568, 49, 1), 0); del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_352, reinterpret_tensor(buf183, (512, 49, 49), (2401, 49, 1), 0), out=buf184)
        del permute_352
        buf185 = buf131; del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf183, (512, 49, 49), (2401, 49, 1), 0), permute_353, out=buf185)
        del permute_353
        buf186 = buf141; del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf185, buf184, buf176, buf186, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf187 = reinterpret_tensor(buf185, (1568, 512), (512, 1), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf186, (1568, 1536), (1536, 1), 0), permute_356, out=buf187)
        del permute_356
        buf188 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf186, (1536, 1568), (1, 1536), 0), view_551, out=buf188)
        del view_551
        buf189 = buf144; del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf186, buf189, 19968, 121, grid=grid(19968), stream=stream0)
        buf190 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_39.run(buf189, buf190, 1536, 13, grid=grid(1536), stream=stream0)
        buf197 = reinterpret_tensor(buf169, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf169  # reuse
        buf198 = reinterpret_tensor(buf184, (8, 196, 512), (100352, 512, 1), 0); del buf184  # reuse
        # Source Nodes: [div__37], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_45.run(buf197, buf187, primals_275, mul_204, div_80, bernoulli_37, buf198, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_37
        del div_80
        del primals_275
        buf193 = reinterpret_tensor(buf173, (512, 13), (1, 512), 0); del buf173  # reuse
        buf195 = buf165; del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_46.run(buf187, mul_204, buf193, buf195, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_204
        buf194 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf193, buf194, 512, 13, grid=grid(512), stream=stream0)
        buf196 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf195, buf196, 512, 13, grid=grid(512), stream=stream0)
        buf199 = reinterpret_tensor(buf158, (1568, 2048), (2048, 1), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf198, (1568, 512), (512, 1), 0), permute_361, out=buf199)
        del permute_361
        buf200 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf198, (512, 1568), (1, 512), 0), view_545, out=buf200)
        del view_545
        buf201 = reinterpret_tensor(buf195, (1, 512, 13), (6656, 1, 512), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf198, buf201, 6656, 121, grid=grid(6656), stream=stream0)
        buf202 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf201, buf202, 512, 13, grid=grid(512), stream=stream0)
        buf203 = reinterpret_tensor(buf199, (8, 196, 2048), (401408, 2048, 1), 0); del buf199  # reuse
        # Source Nodes: [x_369], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_27.run(buf203, addmm_78, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_78
        buf204 = reinterpret_tensor(buf198, (1568, 512), (512, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf203, (1568, 2048), (2048, 1), 0), permute_365, out=buf204)
        del permute_365
        buf205 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf203, (2048, 1568), (1, 2048), 0), view_543, out=buf205)
        del view_543
        buf206 = buf161; del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf203, buf206, 26624, 121, grid=grid(26624), stream=stream0)
        buf207 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf206, buf207, 2048, 13, grid=grid(2048), stream=stream0)
        buf214 = reinterpret_tensor(buf197, (8, 196, 512), (100352, 512, 1), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf214, buf204, primals_269, mul_198, div_81, 1568, 512, grid=grid(1568), stream=stream0)
        del div_81
        del primals_269
        buf210 = reinterpret_tensor(buf201, (512, 13), (1, 512), 0); del buf201  # reuse
        buf212 = buf193; del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_31.run(buf204, mul_198, buf210, buf212, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_198
        buf211 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf210, buf211, 512, 13, grid=grid(512), stream=stream0)
        buf213 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf212, buf213, 512, 13, grid=grid(512), stream=stream0)
        buf215 = reinterpret_tensor(buf204, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_47.run(buf214, bernoulli_36, buf215, 802816, grid=grid(802816), stream=stream0)
        del bernoulli_36
        buf216 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (1568, 512), (512, 1), 0), permute_370, out=buf216)
        del permute_370
        buf217 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (512, 1568), (1, 512), 0), view_537, out=buf217)
        del view_537
        buf218 = reinterpret_tensor(buf212, (1, 512, 13), (6656, 1, 512), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf215, buf218, 6656, 121, grid=grid(6656), stream=stream0)
        buf219 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf218, buf219, 512, 13, grid=grid(512), stream=stream0)
        buf220 = reinterpret_tensor(buf215, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf216, buf220, 802816, grid=grid(802816), stream=stream0)
        buf221 = reinterpret_tensor(buf216, (512, 49, 32), (1568, 32, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_375, reinterpret_tensor(buf220, (512, 49, 32), (1568, 32, 1), 0), out=buf221)
        del permute_375
        buf222 = reinterpret_tensor(buf183, (512, 49, 49), (2401, 49, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf220, (512, 49, 32), (1568, 32, 1), 0), permute_376, out=buf222)
        del permute_376
        buf223 = buf178; del buf178  # reuse
        buf228 = reinterpret_tensor(buf177, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_34.run(buf222, alias_28, buf223, buf228, 25088, 49, grid=grid(25088), stream=stream0)
        buf224 = buf179; del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_35.run(buf222, alias_28, buf223, buf224, 38416, 32, grid=grid(38416), stream=stream0)
        del alias_28
        buf225 = empty((169, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_36.run(buf225, 2704, grid=grid(2704), stream=stream0)
        aten.index_put_(buf225, [view_529], reinterpret_tensor(buf224, (2401, 16), (1, 2401), 0), True)
        del view_529
        buf229 = reinterpret_tensor(buf220, (512, 32, 49), (1568, 49, 1), 0); del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_378, reinterpret_tensor(buf228, (512, 49, 49), (2401, 49, 1), 0), out=buf229)
        del permute_378
        buf230 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf228, (512, 49, 49), (2401, 49, 1), 0), permute_379, out=buf230)
        del permute_379
        buf231 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf230, buf229, buf221, buf231, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf232 = reinterpret_tensor(buf230, (1568, 512), (512, 1), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (1568, 1536), (1536, 1), 0), permute_382, out=buf232)
        del permute_382
        buf233 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (1536, 1568), (1, 1536), 0), view_523, out=buf233)
        del view_523
        buf234 = buf189; del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf231, buf234, 19968, 121, grid=grid(19968), stream=stream0)
        buf235 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_39.run(buf234, buf235, 1536, 13, grid=grid(1536), stream=stream0)
        buf242 = reinterpret_tensor(buf214, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf214  # reuse
        buf243 = reinterpret_tensor(buf229, (8, 196, 512), (100352, 512, 1), 0); del buf229  # reuse
        # Source Nodes: [div__35], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_add_div_mul_native_layer_norm_backward_roll_48.run(buf242, buf232, primals_263, mul_194, div_82, bernoulli_35, buf243, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_35
        del div_82
        del primals_263
        buf238 = reinterpret_tensor(buf218, (512, 13), (13, 1), 0); del buf218  # reuse
        buf240 = reinterpret_tensor(buf210, (512, 13), (13, 1), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_red_fused_native_layer_norm_backward_roll_41.run(buf232, mul_194, buf238, buf240, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_194
        buf239 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_42.run(buf238, buf239, 512, 13, grid=grid(512), stream=stream0)
        buf241 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_42.run(buf240, buf241, 512, 13, grid=grid(512), stream=stream0)
        buf244 = reinterpret_tensor(buf203, (1568, 2048), (2048, 1), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf243, (1568, 512), (512, 1), 0), permute_387, out=buf244)
        del permute_387
        buf245 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf243, (512, 1568), (1, 512), 0), view_517, out=buf245)
        del view_517
        buf246 = reinterpret_tensor(buf240, (1, 512, 13), (6656, 1, 512), 0); del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf243, buf246, 6656, 121, grid=grid(6656), stream=stream0)
        buf247 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf246, buf247, 512, 13, grid=grid(512), stream=stream0)
        buf248 = reinterpret_tensor(buf244, (8, 196, 2048), (401408, 2048, 1), 0); del buf244  # reuse
        # Source Nodes: [x_351], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_27.run(buf248, addmm_74, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_74
        buf249 = reinterpret_tensor(buf243, (1568, 512), (512, 1), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (1568, 2048), (2048, 1), 0), permute_391, out=buf249)
        del permute_391
        buf250 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (2048, 1568), (1, 2048), 0), view_515, out=buf250)
        del view_515
        buf251 = buf206; del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf248, buf251, 26624, 121, grid=grid(26624), stream=stream0)
        buf252 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf251, buf252, 2048, 13, grid=grid(2048), stream=stream0)
        buf259 = reinterpret_tensor(buf242, (8, 196, 512), (100352, 512, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf259, buf249, primals_257, mul_188, div_83, 1568, 512, grid=grid(1568), stream=stream0)
        del div_83
        del primals_257
        buf255 = reinterpret_tensor(buf246, (512, 13), (1, 512), 0); del buf246  # reuse
        buf257 = reinterpret_tensor(buf238, (512, 13), (1, 512), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_31.run(buf249, mul_188, buf255, buf257, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_188
        buf256 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf255, buf256, 512, 13, grid=grid(512), stream=stream0)
        buf258 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf257, buf258, 512, 13, grid=grid(512), stream=stream0)
        buf260 = reinterpret_tensor(buf249, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_49.run(buf259, bernoulli_34, buf260, 802816, grid=grid(802816), stream=stream0)
        del bernoulli_34
        buf261 = buf232; del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf260, (1568, 512), (512, 1), 0), permute_396, out=buf261)
        del permute_396
        buf262 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf260, (512, 1568), (1, 512), 0), view_509, out=buf262)
        del view_509
        buf263 = reinterpret_tensor(buf257, (1, 512, 13), (6656, 1, 512), 0); del buf257  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf260, buf263, 6656, 121, grid=grid(6656), stream=stream0)
        buf264 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf263, buf264, 512, 13, grid=grid(512), stream=stream0)
        buf265 = reinterpret_tensor(buf260, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf261, buf265, 802816, grid=grid(802816), stream=stream0)
        buf266 = reinterpret_tensor(buf261, (512, 49, 32), (1568, 32, 1), 0); del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_401, reinterpret_tensor(buf265, (512, 49, 32), (1568, 32, 1), 0), out=buf266)
        del permute_401
        buf267 = reinterpret_tensor(buf228, (512, 49, 49), (2401, 49, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf265, (512, 49, 32), (1568, 32, 1), 0), permute_402, out=buf267)
        del permute_402
        buf268 = buf223; del buf223  # reuse
        buf273 = reinterpret_tensor(buf222, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_34.run(buf267, alias_29, buf268, buf273, 25088, 49, grid=grid(25088), stream=stream0)
        buf269 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused_sum_35.run(buf267, alias_29, buf268, buf269, 38416, 32, grid=grid(38416), stream=stream0)
        del alias_29
        buf270 = empty((169, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_36.run(buf270, 2704, grid=grid(2704), stream=stream0)
        aten.index_put_(buf270, [view_503], reinterpret_tensor(buf269, (2401, 16), (1, 2401), 0), True)
        del view_503
        buf274 = reinterpret_tensor(buf265, (512, 32, 49), (1568, 49, 1), 0); del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_404, reinterpret_tensor(buf273, (512, 49, 49), (2401, 49, 1), 0), out=buf274)
        del permute_404
        buf275 = buf221; del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf273, (512, 49, 49), (2401, 49, 1), 0), permute_405, out=buf275)
        del permute_405
        buf276 = buf231; del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf275, buf274, buf266, buf276, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf277 = reinterpret_tensor(buf275, (1568, 512), (512, 1), 0); del buf275  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf276, (1568, 1536), (1536, 1), 0), permute_408, out=buf277)
        del permute_408
        buf278 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf276, (1536, 1568), (1, 1536), 0), view_497, out=buf278)
        del view_497
        buf279 = buf234; del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf276, buf279, 19968, 121, grid=grid(19968), stream=stream0)
        buf280 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_39.run(buf279, buf280, 1536, 13, grid=grid(1536), stream=stream0)
        buf287 = reinterpret_tensor(buf259, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf259  # reuse
        buf288 = reinterpret_tensor(buf274, (8, 196, 512), (100352, 512, 1), 0); del buf274  # reuse
        # Source Nodes: [div__33], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_50.run(buf287, buf277, primals_251, mul_184, div_84, bernoulli_33, buf288, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_33
        del div_84
        del primals_251
        buf283 = reinterpret_tensor(buf263, (512, 13), (1, 512), 0); del buf263  # reuse
        buf285 = buf255; del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_46.run(buf277, mul_184, buf283, buf285, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_184
        buf284 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf283, buf284, 512, 13, grid=grid(512), stream=stream0)
        buf286 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf285, buf286, 512, 13, grid=grid(512), stream=stream0)
        buf289 = reinterpret_tensor(buf248, (1568, 2048), (2048, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf288, (1568, 512), (512, 1), 0), permute_413, out=buf289)
        del permute_413
        buf290 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf288, (512, 1568), (1, 512), 0), view_491, out=buf290)
        del view_491
        buf291 = reinterpret_tensor(buf285, (1, 512, 13), (6656, 1, 512), 0); del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf288, buf291, 6656, 121, grid=grid(6656), stream=stream0)
        buf292 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf291, buf292, 512, 13, grid=grid(512), stream=stream0)
        buf293 = reinterpret_tensor(buf289, (8, 196, 2048), (401408, 2048, 1), 0); del buf289  # reuse
        # Source Nodes: [x_333], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_27.run(buf293, addmm_70, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_70
        buf294 = reinterpret_tensor(buf288, (1568, 512), (512, 1), 0); del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (1568, 2048), (2048, 1), 0), permute_417, out=buf294)
        del permute_417
        buf295 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (2048, 1568), (1, 2048), 0), view_489, out=buf295)
        del view_489
        buf296 = buf251; del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf293, buf296, 26624, 121, grid=grid(26624), stream=stream0)
        buf297 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf296, buf297, 2048, 13, grid=grid(2048), stream=stream0)
        buf304 = reinterpret_tensor(buf287, (8, 196, 512), (100352, 512, 1), 0); del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf304, buf294, primals_245, mul_178, div_85, 1568, 512, grid=grid(1568), stream=stream0)
        del div_85
        del primals_245
        buf300 = reinterpret_tensor(buf291, (512, 13), (1, 512), 0); del buf291  # reuse
        buf302 = buf283; del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_31.run(buf294, mul_178, buf300, buf302, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_178
        buf301 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf300, buf301, 512, 13, grid=grid(512), stream=stream0)
        buf303 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf302, buf303, 512, 13, grid=grid(512), stream=stream0)
        buf305 = reinterpret_tensor(buf294, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_51.run(buf304, bernoulli_32, buf305, 802816, grid=grid(802816), stream=stream0)
        del bernoulli_32
        buf306 = buf277; del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf305, (1568, 512), (512, 1), 0), permute_422, out=buf306)
        del permute_422
        buf307 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf305, (512, 1568), (1, 512), 0), view_483, out=buf307)
        del view_483
        buf308 = reinterpret_tensor(buf302, (1, 512, 13), (6656, 1, 512), 0); del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf305, buf308, 6656, 121, grid=grid(6656), stream=stream0)
        buf309 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf308, buf309, 512, 13, grid=grid(512), stream=stream0)
        buf310 = reinterpret_tensor(buf305, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf306, buf310, 802816, grid=grid(802816), stream=stream0)
        buf311 = reinterpret_tensor(buf306, (512, 49, 32), (1568, 32, 1), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_427, reinterpret_tensor(buf310, (512, 49, 32), (1568, 32, 1), 0), out=buf311)
        del permute_427
        buf312 = reinterpret_tensor(buf273, (512, 49, 49), (2401, 49, 1), 0); del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf310, (512, 49, 32), (1568, 32, 1), 0), permute_428, out=buf312)
        del permute_428
        buf313 = buf268; del buf268  # reuse
        buf318 = reinterpret_tensor(buf267, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_34.run(buf312, alias_30, buf313, buf318, 25088, 49, grid=grid(25088), stream=stream0)
        buf314 = buf269; del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_35.run(buf312, alias_30, buf313, buf314, 38416, 32, grid=grid(38416), stream=stream0)
        del alias_30
        buf315 = empty((169, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_36.run(buf315, 2704, grid=grid(2704), stream=stream0)
        aten.index_put_(buf315, [view_475], reinterpret_tensor(buf314, (2401, 16), (1, 2401), 0), True)
        del view_475
        buf319 = reinterpret_tensor(buf310, (512, 32, 49), (1568, 49, 1), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_430, reinterpret_tensor(buf318, (512, 49, 49), (2401, 49, 1), 0), out=buf319)
        del permute_430
        buf320 = buf266; del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf318, (512, 49, 49), (2401, 49, 1), 0), permute_431, out=buf320)
        del permute_431
        buf321 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf320, buf319, buf311, buf321, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf322 = reinterpret_tensor(buf320, (1568, 512), (512, 1), 0); del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf321, (1568, 1536), (1536, 1), 0), permute_434, out=buf322)
        del permute_434
        buf323 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf321, (1536, 1568), (1, 1536), 0), view_469, out=buf323)
        del view_469
        buf324 = buf279; del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf321, buf324, 19968, 121, grid=grid(19968), stream=stream0)
        buf325 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_39.run(buf324, buf325, 1536, 13, grid=grid(1536), stream=stream0)
        buf332 = reinterpret_tensor(buf304, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf304  # reuse
        buf333 = reinterpret_tensor(buf319, (8, 196, 512), (100352, 512, 1), 0); del buf319  # reuse
        # Source Nodes: [div__31], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_add_div_mul_native_layer_norm_backward_roll_52.run(buf332, buf322, primals_239, mul_174, div_86, bernoulli_31, buf333, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_31
        del div_86
        del primals_239
        buf328 = reinterpret_tensor(buf308, (512, 13), (13, 1), 0); del buf308  # reuse
        buf330 = reinterpret_tensor(buf300, (512, 13), (13, 1), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_red_fused_native_layer_norm_backward_roll_41.run(buf322, mul_174, buf328, buf330, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_174
        buf329 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_42.run(buf328, buf329, 512, 13, grid=grid(512), stream=stream0)
        buf331 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_42.run(buf330, buf331, 512, 13, grid=grid(512), stream=stream0)
        buf334 = reinterpret_tensor(buf293, (1568, 2048), (2048, 1), 0); del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (1568, 512), (512, 1), 0), permute_439, out=buf334)
        del permute_439
        buf335 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (512, 1568), (1, 512), 0), view_463, out=buf335)
        del view_463
        buf336 = reinterpret_tensor(buf330, (1, 512, 13), (6656, 1, 512), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf333, buf336, 6656, 121, grid=grid(6656), stream=stream0)
        buf337 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf336, buf337, 512, 13, grid=grid(512), stream=stream0)
        buf338 = reinterpret_tensor(buf334, (8, 196, 2048), (401408, 2048, 1), 0); del buf334  # reuse
        # Source Nodes: [x_315], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_27.run(buf338, addmm_66, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_66
        buf339 = reinterpret_tensor(buf333, (1568, 512), (512, 1), 0); del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (1568, 2048), (2048, 1), 0), permute_443, out=buf339)
        del permute_443
        buf340 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (2048, 1568), (1, 2048), 0), view_461, out=buf340)
        del view_461
        buf341 = buf296; del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf338, buf341, 26624, 121, grid=grid(26624), stream=stream0)
        buf342 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf341, buf342, 2048, 13, grid=grid(2048), stream=stream0)
        buf349 = reinterpret_tensor(buf332, (8, 196, 512), (100352, 512, 1), 0); del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf349, buf339, primals_233, mul_168, div_87, 1568, 512, grid=grid(1568), stream=stream0)
        del div_87
        del primals_233
        buf345 = reinterpret_tensor(buf336, (512, 13), (1, 512), 0); del buf336  # reuse
        buf347 = reinterpret_tensor(buf328, (512, 13), (1, 512), 0); del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_31.run(buf339, mul_168, buf345, buf347, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_168
        buf346 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf345, buf346, 512, 13, grid=grid(512), stream=stream0)
        buf348 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf347, buf348, 512, 13, grid=grid(512), stream=stream0)
        buf350 = reinterpret_tensor(buf339, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_53.run(buf349, bernoulli_30, buf350, 802816, grid=grid(802816), stream=stream0)
        del bernoulli_30
        buf351 = buf322; del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf350, (1568, 512), (512, 1), 0), permute_448, out=buf351)
        del permute_448
        buf352 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf350, (512, 1568), (1, 512), 0), view_455, out=buf352)
        del view_455
        buf353 = reinterpret_tensor(buf347, (1, 512, 13), (6656, 1, 512), 0); del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf350, buf353, 6656, 121, grid=grid(6656), stream=stream0)
        buf354 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf353, buf354, 512, 13, grid=grid(512), stream=stream0)
        buf355 = reinterpret_tensor(buf350, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf351, buf355, 802816, grid=grid(802816), stream=stream0)
        buf356 = reinterpret_tensor(buf351, (512, 49, 32), (1568, 32, 1), 0); del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_453, reinterpret_tensor(buf355, (512, 49, 32), (1568, 32, 1), 0), out=buf356)
        del permute_453
        buf357 = reinterpret_tensor(buf318, (512, 49, 49), (2401, 49, 1), 0); del buf318  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf355, (512, 49, 32), (1568, 32, 1), 0), permute_454, out=buf357)
        del permute_454
        buf358 = buf313; del buf313  # reuse
        buf363 = reinterpret_tensor(buf312, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_34.run(buf357, alias_31, buf358, buf363, 25088, 49, grid=grid(25088), stream=stream0)
        buf359 = buf314; del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused_sum_35.run(buf357, alias_31, buf358, buf359, 38416, 32, grid=grid(38416), stream=stream0)
        del alias_31
        buf360 = empty((169, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_36.run(buf360, 2704, grid=grid(2704), stream=stream0)
        aten.index_put_(buf360, [view_449], reinterpret_tensor(buf359, (2401, 16), (1, 2401), 0), True)
        del view_449
        buf364 = reinterpret_tensor(buf355, (512, 32, 49), (1568, 49, 1), 0); del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_456, reinterpret_tensor(buf363, (512, 49, 49), (2401, 49, 1), 0), out=buf364)
        del permute_456
        buf365 = buf311; del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf363, (512, 49, 49), (2401, 49, 1), 0), permute_457, out=buf365)
        del permute_457
        buf366 = buf321; del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf365, buf364, buf356, buf366, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf367 = reinterpret_tensor(buf365, (1568, 512), (512, 1), 0); del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf366, (1568, 1536), (1536, 1), 0), permute_460, out=buf367)
        del permute_460
        buf368 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf366, (1536, 1568), (1, 1536), 0), view_443, out=buf368)
        del view_443
        buf369 = buf324; del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf366, buf369, 19968, 121, grid=grid(19968), stream=stream0)
        buf370 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_39.run(buf369, buf370, 1536, 13, grid=grid(1536), stream=stream0)
        buf377 = reinterpret_tensor(buf349, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf349  # reuse
        buf378 = reinterpret_tensor(buf364, (8, 196, 512), (100352, 512, 1), 0); del buf364  # reuse
        # Source Nodes: [div__29], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_54.run(buf377, buf367, primals_227, mul_164, div_88, bernoulli_29, buf378, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_29
        del div_88
        del primals_227
        buf373 = reinterpret_tensor(buf353, (512, 13), (1, 512), 0); del buf353  # reuse
        buf375 = buf345; del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_46.run(buf367, mul_164, buf373, buf375, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_164
        buf374 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf373, buf374, 512, 13, grid=grid(512), stream=stream0)
        buf376 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf375, buf376, 512, 13, grid=grid(512), stream=stream0)
        buf379 = reinterpret_tensor(buf338, (1568, 2048), (2048, 1), 0); del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf378, (1568, 512), (512, 1), 0), permute_465, out=buf379)
        del permute_465
        buf380 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf378, (512, 1568), (1, 512), 0), view_437, out=buf380)
        del view_437
        buf381 = reinterpret_tensor(buf375, (1, 512, 13), (6656, 1, 512), 0); del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf378, buf381, 6656, 121, grid=grid(6656), stream=stream0)
        buf382 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf381, buf382, 512, 13, grid=grid(512), stream=stream0)
        buf383 = reinterpret_tensor(buf379, (8, 196, 2048), (401408, 2048, 1), 0); del buf379  # reuse
        # Source Nodes: [x_297], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_27.run(buf383, addmm_62, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_62
        buf384 = reinterpret_tensor(buf378, (1568, 512), (512, 1), 0); del buf378  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf383, (1568, 2048), (2048, 1), 0), permute_469, out=buf384)
        del permute_469
        buf385 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf383, (2048, 1568), (1, 2048), 0), view_435, out=buf385)
        del view_435
        buf386 = buf341; del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf383, buf386, 26624, 121, grid=grid(26624), stream=stream0)
        buf387 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf386, buf387, 2048, 13, grid=grid(2048), stream=stream0)
        buf394 = reinterpret_tensor(buf377, (8, 196, 512), (100352, 512, 1), 0); del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf394, buf384, primals_221, mul_158, div_89, 1568, 512, grid=grid(1568), stream=stream0)
        del div_89
        del primals_221
        buf390 = reinterpret_tensor(buf381, (512, 13), (1, 512), 0); del buf381  # reuse
        buf392 = buf373; del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_31.run(buf384, mul_158, buf390, buf392, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_158
        buf391 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf390, buf391, 512, 13, grid=grid(512), stream=stream0)
        buf393 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf392, buf393, 512, 13, grid=grid(512), stream=stream0)
        buf395 = reinterpret_tensor(buf384, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_55.run(buf394, bernoulli_28, buf395, 802816, grid=grid(802816), stream=stream0)
        del bernoulli_28
        buf396 = buf367; del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf395, (1568, 512), (512, 1), 0), permute_474, out=buf396)
        del permute_474
        buf397 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf395, (512, 1568), (1, 512), 0), view_429, out=buf397)
        del view_429
        buf398 = reinterpret_tensor(buf392, (1, 512, 13), (6656, 1, 512), 0); del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf395, buf398, 6656, 121, grid=grid(6656), stream=stream0)
        buf399 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf398, buf399, 512, 13, grid=grid(512), stream=stream0)
        buf400 = reinterpret_tensor(buf395, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf396, buf400, 802816, grid=grid(802816), stream=stream0)
        buf401 = reinterpret_tensor(buf396, (512, 49, 32), (1568, 32, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_479, reinterpret_tensor(buf400, (512, 49, 32), (1568, 32, 1), 0), out=buf401)
        del permute_479
        buf402 = reinterpret_tensor(buf363, (512, 49, 49), (2401, 49, 1), 0); del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf400, (512, 49, 32), (1568, 32, 1), 0), permute_480, out=buf402)
        del permute_480
        buf403 = buf358; del buf358  # reuse
        buf408 = reinterpret_tensor(buf357, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_34.run(buf402, alias_32, buf403, buf408, 25088, 49, grid=grid(25088), stream=stream0)
        buf404 = buf359; del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_35.run(buf402, alias_32, buf403, buf404, 38416, 32, grid=grid(38416), stream=stream0)
        del alias_32
        buf405 = empty((169, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_36.run(buf405, 2704, grid=grid(2704), stream=stream0)
        aten.index_put_(buf405, [view_421], reinterpret_tensor(buf404, (2401, 16), (1, 2401), 0), True)
        del view_421
        buf409 = reinterpret_tensor(buf400, (512, 32, 49), (1568, 49, 1), 0); del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_482, reinterpret_tensor(buf408, (512, 49, 49), (2401, 49, 1), 0), out=buf409)
        del permute_482
        buf410 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf408, (512, 49, 49), (2401, 49, 1), 0), permute_483, out=buf410)
        del permute_483
        buf411 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf410, buf409, buf401, buf411, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf412 = reinterpret_tensor(buf410, (1568, 512), (512, 1), 0); del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf411, (1568, 1536), (1536, 1), 0), permute_486, out=buf412)
        del permute_486
        buf413 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf411, (1536, 1568), (1, 1536), 0), view_415, out=buf413)
        del view_415
        buf414 = buf369; del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf411, buf414, 19968, 121, grid=grid(19968), stream=stream0)
        buf415 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_39.run(buf414, buf415, 1536, 13, grid=grid(1536), stream=stream0)
        buf422 = reinterpret_tensor(buf394, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf394  # reuse
        buf423 = reinterpret_tensor(buf409, (8, 196, 512), (100352, 512, 1), 0); del buf409  # reuse
        # Source Nodes: [div__27], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_add_div_mul_native_layer_norm_backward_roll_56.run(buf422, buf412, primals_215, mul_154, div_90, bernoulli_27, buf423, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_27
        del div_90
        del primals_215
        buf418 = reinterpret_tensor(buf398, (512, 13), (13, 1), 0); del buf398  # reuse
        buf420 = reinterpret_tensor(buf390, (512, 13), (13, 1), 0); del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_red_fused_native_layer_norm_backward_roll_41.run(buf412, mul_154, buf418, buf420, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_154
        buf419 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_42.run(buf418, buf419, 512, 13, grid=grid(512), stream=stream0)
        buf421 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_42.run(buf420, buf421, 512, 13, grid=grid(512), stream=stream0)
        buf424 = reinterpret_tensor(buf383, (1568, 2048), (2048, 1), 0); del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf423, (1568, 512), (512, 1), 0), permute_491, out=buf424)
        del permute_491
        buf425 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf423, (512, 1568), (1, 512), 0), view_409, out=buf425)
        del view_409
        buf426 = reinterpret_tensor(buf420, (1, 512, 13), (6656, 1, 512), 0); del buf420  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf423, buf426, 6656, 121, grid=grid(6656), stream=stream0)
        buf427 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf426, buf427, 512, 13, grid=grid(512), stream=stream0)
        buf428 = reinterpret_tensor(buf424, (8, 196, 2048), (401408, 2048, 1), 0); del buf424  # reuse
        # Source Nodes: [x_279], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_27.run(buf428, addmm_58, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_58
        buf429 = reinterpret_tensor(buf423, (1568, 512), (512, 1), 0); del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (1568, 2048), (2048, 1), 0), permute_495, out=buf429)
        del permute_495
        buf430 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (2048, 1568), (1, 2048), 0), view_407, out=buf430)
        del view_407
        buf431 = buf386; del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf428, buf431, 26624, 121, grid=grid(26624), stream=stream0)
        buf432 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf431, buf432, 2048, 13, grid=grid(2048), stream=stream0)
        buf439 = reinterpret_tensor(buf422, (8, 196, 512), (100352, 512, 1), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf439, buf429, primals_209, mul_148, div_91, 1568, 512, grid=grid(1568), stream=stream0)
        del div_91
        del primals_209
        buf435 = reinterpret_tensor(buf426, (512, 13), (1, 512), 0); del buf426  # reuse
        buf437 = reinterpret_tensor(buf418, (512, 13), (1, 512), 0); del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_31.run(buf429, mul_148, buf435, buf437, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_148
        buf436 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf435, buf436, 512, 13, grid=grid(512), stream=stream0)
        buf438 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf437, buf438, 512, 13, grid=grid(512), stream=stream0)
        buf440 = reinterpret_tensor(buf429, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_57.run(buf439, bernoulli_26, buf440, 802816, grid=grid(802816), stream=stream0)
        del bernoulli_26
        buf441 = buf412; del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf440, (1568, 512), (512, 1), 0), permute_500, out=buf441)
        del permute_500
        buf442 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf440, (512, 1568), (1, 512), 0), view_401, out=buf442)
        del view_401
        buf443 = reinterpret_tensor(buf437, (1, 512, 13), (6656, 1, 512), 0); del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf440, buf443, 6656, 121, grid=grid(6656), stream=stream0)
        buf444 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf443, buf444, 512, 13, grid=grid(512), stream=stream0)
        buf445 = reinterpret_tensor(buf440, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf440  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf441, buf445, 802816, grid=grid(802816), stream=stream0)
        buf446 = reinterpret_tensor(buf441, (512, 49, 32), (1568, 32, 1), 0); del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_505, reinterpret_tensor(buf445, (512, 49, 32), (1568, 32, 1), 0), out=buf446)
        del permute_505
        buf447 = reinterpret_tensor(buf408, (512, 49, 49), (2401, 49, 1), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf445, (512, 49, 32), (1568, 32, 1), 0), permute_506, out=buf447)
        del permute_506
        buf448 = buf403; del buf403  # reuse
        buf453 = reinterpret_tensor(buf402, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_34.run(buf447, alias_33, buf448, buf453, 25088, 49, grid=grid(25088), stream=stream0)
        buf449 = buf404; del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused_sum_35.run(buf447, alias_33, buf448, buf449, 38416, 32, grid=grid(38416), stream=stream0)
        del alias_33
        buf450 = empty((169, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_36.run(buf450, 2704, grid=grid(2704), stream=stream0)
        aten.index_put_(buf450, [view_395], reinterpret_tensor(buf449, (2401, 16), (1, 2401), 0), True)
        del view_395
        buf454 = reinterpret_tensor(buf445, (512, 32, 49), (1568, 49, 1), 0); del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_508, reinterpret_tensor(buf453, (512, 49, 49), (2401, 49, 1), 0), out=buf454)
        del permute_508
        buf455 = buf401; del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf453, (512, 49, 49), (2401, 49, 1), 0), permute_509, out=buf455)
        del permute_509
        buf456 = buf411; del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf455, buf454, buf446, buf456, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf457 = reinterpret_tensor(buf455, (1568, 512), (512, 1), 0); del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf456, (1568, 1536), (1536, 1), 0), permute_512, out=buf457)
        del permute_512
        buf458 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf456, (1536, 1568), (1, 1536), 0), view_389, out=buf458)
        del view_389
        buf459 = buf414; del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf456, buf459, 19968, 121, grid=grid(19968), stream=stream0)
        buf460 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_39.run(buf459, buf460, 1536, 13, grid=grid(1536), stream=stream0)
        buf467 = reinterpret_tensor(buf439, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf439  # reuse
        buf468 = reinterpret_tensor(buf454, (8, 196, 512), (100352, 512, 1), 0); del buf454  # reuse
        # Source Nodes: [div__25], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_58.run(buf467, buf457, primals_203, mul_144, div_92, bernoulli_25, buf468, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_25
        del div_92
        del primals_203
        buf463 = reinterpret_tensor(buf443, (512, 13), (1, 512), 0); del buf443  # reuse
        buf465 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_46.run(buf457, mul_144, buf463, buf465, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_144
        buf464 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf463, buf464, 512, 13, grid=grid(512), stream=stream0)
        buf466 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf465, buf466, 512, 13, grid=grid(512), stream=stream0)
        buf469 = reinterpret_tensor(buf428, (1568, 2048), (2048, 1), 0); del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf468, (1568, 512), (512, 1), 0), permute_517, out=buf469)
        del permute_517
        buf470 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf468, (512, 1568), (1, 512), 0), view_383, out=buf470)
        del view_383
        buf471 = reinterpret_tensor(buf465, (1, 512, 13), (6656, 1, 512), 0); del buf465  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf468, buf471, 6656, 121, grid=grid(6656), stream=stream0)
        buf472 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf471, buf472, 512, 13, grid=grid(512), stream=stream0)
        buf473 = reinterpret_tensor(buf469, (8, 196, 2048), (401408, 2048, 1), 0); del buf469  # reuse
        # Source Nodes: [x_261], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_27.run(buf473, addmm_54, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_54
        buf474 = reinterpret_tensor(buf468, (1568, 512), (512, 1), 0); del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf473, (1568, 2048), (2048, 1), 0), permute_521, out=buf474)
        del permute_521
        buf475 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf473, (2048, 1568), (1, 2048), 0), view_381, out=buf475)
        del view_381
        buf476 = buf431; del buf431  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf473, buf476, 26624, 121, grid=grid(26624), stream=stream0)
        buf477 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf476, buf477, 2048, 13, grid=grid(2048), stream=stream0)
        buf484 = reinterpret_tensor(buf467, (8, 196, 512), (100352, 512, 1), 0); del buf467  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf484, buf474, primals_197, mul_138, div_93, 1568, 512, grid=grid(1568), stream=stream0)
        del div_93
        del primals_197
        buf480 = reinterpret_tensor(buf471, (512, 13), (1, 512), 0); del buf471  # reuse
        buf482 = buf463; del buf463  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_31.run(buf474, mul_138, buf480, buf482, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_138
        buf481 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf480, buf481, 512, 13, grid=grid(512), stream=stream0)
        buf483 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf482, buf483, 512, 13, grid=grid(512), stream=stream0)
        buf485 = reinterpret_tensor(buf474, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_59.run(buf484, bernoulli_24, buf485, 802816, grid=grid(802816), stream=stream0)
        del bernoulli_24
        buf486 = buf457; del buf457  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf485, (1568, 512), (512, 1), 0), permute_526, out=buf486)
        del permute_526
        buf487 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf485, (512, 1568), (1, 512), 0), view_375, out=buf487)
        del view_375
        buf488 = reinterpret_tensor(buf482, (1, 512, 13), (6656, 1, 512), 0); del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf485, buf488, 6656, 121, grid=grid(6656), stream=stream0)
        buf489 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf488, buf489, 512, 13, grid=grid(512), stream=stream0)
        buf490 = reinterpret_tensor(buf485, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf486, buf490, 802816, grid=grid(802816), stream=stream0)
        buf491 = reinterpret_tensor(buf486, (512, 49, 32), (1568, 32, 1), 0); del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_531, reinterpret_tensor(buf490, (512, 49, 32), (1568, 32, 1), 0), out=buf491)
        del permute_531
        buf492 = reinterpret_tensor(buf453, (512, 49, 49), (2401, 49, 1), 0); del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf490, (512, 49, 32), (1568, 32, 1), 0), permute_532, out=buf492)
        del permute_532
        buf493 = buf448; del buf448  # reuse
        buf498 = reinterpret_tensor(buf447, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_34.run(buf492, alias_34, buf493, buf498, 25088, 49, grid=grid(25088), stream=stream0)
        buf494 = buf449; del buf449  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_35.run(buf492, alias_34, buf493, buf494, 38416, 32, grid=grid(38416), stream=stream0)
        del alias_34
        buf495 = empty((169, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_36.run(buf495, 2704, grid=grid(2704), stream=stream0)
        aten.index_put_(buf495, [view_367], reinterpret_tensor(buf494, (2401, 16), (1, 2401), 0), True)
        del view_367
        buf499 = reinterpret_tensor(buf490, (512, 32, 49), (1568, 49, 1), 0); del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_534, reinterpret_tensor(buf498, (512, 49, 49), (2401, 49, 1), 0), out=buf499)
        del permute_534
        buf500 = buf446; del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf498, (512, 49, 49), (2401, 49, 1), 0), permute_535, out=buf500)
        del permute_535
        buf501 = buf456; del buf456  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf500, buf499, buf491, buf501, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf502 = reinterpret_tensor(buf500, (1568, 512), (512, 1), 0); del buf500  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf501, (1568, 1536), (1536, 1), 0), permute_538, out=buf502)
        del permute_538
        buf503 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf501, (1536, 1568), (1, 1536), 0), view_361, out=buf503)
        del view_361
        buf504 = buf459; del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf501, buf504, 19968, 121, grid=grid(19968), stream=stream0)
        buf505 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_39.run(buf504, buf505, 1536, 13, grid=grid(1536), stream=stream0)
        buf512 = reinterpret_tensor(buf484, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf484  # reuse
        buf513 = reinterpret_tensor(buf499, (8, 196, 512), (100352, 512, 1), 0); del buf499  # reuse
        # Source Nodes: [div__23], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_add_div_mul_native_layer_norm_backward_roll_60.run(buf512, buf502, primals_191, mul_134, div_94, bernoulli_23, buf513, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_23
        del div_94
        del primals_191
        buf508 = reinterpret_tensor(buf488, (512, 13), (13, 1), 0); del buf488  # reuse
        buf510 = reinterpret_tensor(buf480, (512, 13), (13, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_red_fused_native_layer_norm_backward_roll_41.run(buf502, mul_134, buf508, buf510, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_134
        buf509 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_42.run(buf508, buf509, 512, 13, grid=grid(512), stream=stream0)
        buf511 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_42.run(buf510, buf511, 512, 13, grid=grid(512), stream=stream0)
        buf514 = reinterpret_tensor(buf473, (1568, 2048), (2048, 1), 0); del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf513, (1568, 512), (512, 1), 0), permute_543, out=buf514)
        del permute_543
        buf515 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf513, (512, 1568), (1, 512), 0), view_355, out=buf515)
        del view_355
        buf516 = reinterpret_tensor(buf510, (1, 512, 13), (6656, 1, 512), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf513, buf516, 6656, 121, grid=grid(6656), stream=stream0)
        buf517 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf516, buf517, 512, 13, grid=grid(512), stream=stream0)
        buf518 = reinterpret_tensor(buf514, (8, 196, 2048), (401408, 2048, 1), 0); del buf514  # reuse
        # Source Nodes: [x_243], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_27.run(buf518, addmm_50, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_50
        buf519 = reinterpret_tensor(buf513, (1568, 512), (512, 1), 0); del buf513  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf518, (1568, 2048), (2048, 1), 0), permute_547, out=buf519)
        del permute_547
        buf520 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf518, (2048, 1568), (1, 2048), 0), view_353, out=buf520)
        del view_353
        buf521 = buf476; del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf518, buf521, 26624, 121, grid=grid(26624), stream=stream0)
        buf522 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf521, buf522, 2048, 13, grid=grid(2048), stream=stream0)
        buf529 = reinterpret_tensor(buf512, (8, 196, 512), (100352, 512, 1), 0); del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf529, buf519, primals_185, mul_128, div_95, 1568, 512, grid=grid(1568), stream=stream0)
        del div_95
        del primals_185
        buf525 = reinterpret_tensor(buf516, (512, 13), (1, 512), 0); del buf516  # reuse
        buf527 = reinterpret_tensor(buf508, (512, 13), (1, 512), 0); del buf508  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_31.run(buf519, mul_128, buf525, buf527, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_128
        buf526 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf525, buf526, 512, 13, grid=grid(512), stream=stream0)
        buf528 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf527, buf528, 512, 13, grid=grid(512), stream=stream0)
        buf530 = reinterpret_tensor(buf519, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf519  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_61.run(buf529, bernoulli_22, buf530, 802816, grid=grid(802816), stream=stream0)
        del bernoulli_22
        buf531 = buf502; del buf502  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf530, (1568, 512), (512, 1), 0), permute_552, out=buf531)
        del permute_552
        buf532 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf530, (512, 1568), (1, 512), 0), view_347, out=buf532)
        del view_347
        buf533 = reinterpret_tensor(buf527, (1, 512, 13), (6656, 1, 512), 0); del buf527  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf530, buf533, 6656, 121, grid=grid(6656), stream=stream0)
        buf534 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf533, buf534, 512, 13, grid=grid(512), stream=stream0)
        buf535 = reinterpret_tensor(buf530, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf531, buf535, 802816, grid=grid(802816), stream=stream0)
        buf536 = reinterpret_tensor(buf531, (512, 49, 32), (1568, 32, 1), 0); del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_557, reinterpret_tensor(buf535, (512, 49, 32), (1568, 32, 1), 0), out=buf536)
        del permute_557
        buf537 = reinterpret_tensor(buf498, (512, 49, 49), (2401, 49, 1), 0); del buf498  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf535, (512, 49, 32), (1568, 32, 1), 0), permute_558, out=buf537)
        del permute_558
        buf538 = buf493; del buf493  # reuse
        buf543 = reinterpret_tensor(buf492, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf492  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_34.run(buf537, alias_35, buf538, buf543, 25088, 49, grid=grid(25088), stream=stream0)
        buf539 = buf494; del buf494  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused_sum_35.run(buf537, alias_35, buf538, buf539, 38416, 32, grid=grid(38416), stream=stream0)
        del alias_35
        buf540 = empty((169, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_36.run(buf540, 2704, grid=grid(2704), stream=stream0)
        aten.index_put_(buf540, [view_341], reinterpret_tensor(buf539, (2401, 16), (1, 2401), 0), True)
        del view_341
        buf544 = reinterpret_tensor(buf535, (512, 32, 49), (1568, 49, 1), 0); del buf535  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_560, reinterpret_tensor(buf543, (512, 49, 49), (2401, 49, 1), 0), out=buf544)
        del permute_560
        buf545 = buf491; del buf491  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf543, (512, 49, 49), (2401, 49, 1), 0), permute_561, out=buf545)
        del permute_561
        buf546 = buf501; del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf545, buf544, buf536, buf546, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf547 = reinterpret_tensor(buf545, (1568, 512), (512, 1), 0); del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf546, (1568, 1536), (1536, 1), 0), permute_564, out=buf547)
        del permute_564
        buf548 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf546, (1536, 1568), (1, 1536), 0), view_335, out=buf548)
        del view_335
        buf549 = buf504; del buf504  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf546, buf549, 19968, 121, grid=grid(19968), stream=stream0)
        buf550 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_39.run(buf549, buf550, 1536, 13, grid=grid(1536), stream=stream0)
        buf557 = reinterpret_tensor(buf529, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf529  # reuse
        buf558 = reinterpret_tensor(buf544, (8, 196, 512), (100352, 512, 1), 0); del buf544  # reuse
        # Source Nodes: [div__21], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_62.run(buf557, buf547, primals_179, mul_124, div_96, bernoulli_21, buf558, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_21
        del div_96
        del primals_179
        buf553 = reinterpret_tensor(buf533, (512, 13), (1, 512), 0); del buf533  # reuse
        buf555 = buf525; del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_46.run(buf547, mul_124, buf553, buf555, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_124
        buf554 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf553, buf554, 512, 13, grid=grid(512), stream=stream0)
        buf556 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf555, buf556, 512, 13, grid=grid(512), stream=stream0)
        buf559 = reinterpret_tensor(buf518, (1568, 2048), (2048, 1), 0); del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf558, (1568, 512), (512, 1), 0), permute_569, out=buf559)
        del permute_569
        buf560 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf558, (512, 1568), (1, 512), 0), view_329, out=buf560)
        del view_329
        buf561 = reinterpret_tensor(buf555, (1, 512, 13), (6656, 1, 512), 0); del buf555  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf558, buf561, 6656, 121, grid=grid(6656), stream=stream0)
        buf562 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf561, buf562, 512, 13, grid=grid(512), stream=stream0)
        buf563 = reinterpret_tensor(buf559, (8, 196, 2048), (401408, 2048, 1), 0); del buf559  # reuse
        # Source Nodes: [x_225], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_27.run(buf563, addmm_46, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_46
        buf564 = reinterpret_tensor(buf558, (1568, 512), (512, 1), 0); del buf558  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf563, (1568, 2048), (2048, 1), 0), permute_573, out=buf564)
        del permute_573
        buf565 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf563, (2048, 1568), (1, 2048), 0), view_327, out=buf565)
        del view_327
        buf566 = buf521; del buf521  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf563, buf566, 26624, 121, grid=grid(26624), stream=stream0)
        buf567 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf566, buf567, 2048, 13, grid=grid(2048), stream=stream0)
        buf574 = reinterpret_tensor(buf557, (8, 196, 512), (100352, 512, 1), 0); del buf557  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf574, buf564, primals_173, mul_118, div_97, 1568, 512, grid=grid(1568), stream=stream0)
        del div_97
        del primals_173
        buf570 = reinterpret_tensor(buf561, (512, 13), (1, 512), 0); del buf561  # reuse
        buf572 = buf553; del buf553  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_31.run(buf564, mul_118, buf570, buf572, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_118
        buf571 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf570, buf571, 512, 13, grid=grid(512), stream=stream0)
        buf573 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf572, buf573, 512, 13, grid=grid(512), stream=stream0)
        buf575 = reinterpret_tensor(buf564, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf564  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_63.run(buf574, bernoulli_20, buf575, 802816, grid=grid(802816), stream=stream0)
        del bernoulli_20
        buf576 = buf547; del buf547  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf575, (1568, 512), (512, 1), 0), permute_578, out=buf576)
        del permute_578
        buf577 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf575, (512, 1568), (1, 512), 0), view_321, out=buf577)
        del view_321
        buf578 = reinterpret_tensor(buf572, (1, 512, 13), (6656, 1, 512), 0); del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf575, buf578, 6656, 121, grid=grid(6656), stream=stream0)
        buf579 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf578, buf579, 512, 13, grid=grid(512), stream=stream0)
        buf580 = reinterpret_tensor(buf575, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf575  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf576, buf580, 802816, grid=grid(802816), stream=stream0)
        buf581 = reinterpret_tensor(buf576, (512, 49, 32), (1568, 32, 1), 0); del buf576  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_583, reinterpret_tensor(buf580, (512, 49, 32), (1568, 32, 1), 0), out=buf581)
        del permute_583
        buf582 = reinterpret_tensor(buf543, (512, 49, 49), (2401, 49, 1), 0); del buf543  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf580, (512, 49, 32), (1568, 32, 1), 0), permute_584, out=buf582)
        del permute_584
        buf583 = buf538; del buf538  # reuse
        buf588 = reinterpret_tensor(buf537, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf537  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_34.run(buf582, alias_36, buf583, buf588, 25088, 49, grid=grid(25088), stream=stream0)
        buf584 = buf539; del buf539  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_35.run(buf582, alias_36, buf583, buf584, 38416, 32, grid=grid(38416), stream=stream0)
        del alias_36
        buf585 = empty((169, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_36.run(buf585, 2704, grid=grid(2704), stream=stream0)
        aten.index_put_(buf585, [view_313], reinterpret_tensor(buf584, (2401, 16), (1, 2401), 0), True)
        del view_313
        buf589 = reinterpret_tensor(buf580, (512, 32, 49), (1568, 49, 1), 0); del buf580  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_586, reinterpret_tensor(buf588, (512, 49, 49), (2401, 49, 1), 0), out=buf589)
        del permute_586
        buf590 = buf536; del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf588, (512, 49, 49), (2401, 49, 1), 0), permute_587, out=buf590)
        del permute_587
        buf591 = buf546; del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf590, buf589, buf581, buf591, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf592 = reinterpret_tensor(buf590, (1568, 512), (512, 1), 0); del buf590  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf591, (1568, 1536), (1536, 1), 0), permute_590, out=buf592)
        del permute_590
        buf593 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf591, (1536, 1568), (1, 1536), 0), view_307, out=buf593)
        del view_307
        buf594 = buf549; del buf549  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf591, buf594, 19968, 121, grid=grid(19968), stream=stream0)
        buf595 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_39.run(buf594, buf595, 1536, 13, grid=grid(1536), stream=stream0)
        buf602 = reinterpret_tensor(buf574, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf574  # reuse
        buf603 = reinterpret_tensor(buf589, (8, 196, 512), (100352, 512, 1), 0); del buf589  # reuse
        # Source Nodes: [div__19], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_add_div_mul_native_layer_norm_backward_roll_64.run(buf602, buf592, primals_167, mul_114, div_98, bernoulli_19, buf603, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_19
        del div_98
        del primals_167
        buf598 = reinterpret_tensor(buf578, (512, 13), (13, 1), 0); del buf578  # reuse
        buf600 = reinterpret_tensor(buf570, (512, 13), (13, 1), 0); del buf570  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_red_fused_native_layer_norm_backward_roll_41.run(buf592, mul_114, buf598, buf600, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_114
        buf599 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_42.run(buf598, buf599, 512, 13, grid=grid(512), stream=stream0)
        buf601 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_42.run(buf600, buf601, 512, 13, grid=grid(512), stream=stream0)
        buf604 = reinterpret_tensor(buf563, (1568, 2048), (2048, 1), 0); del buf563  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf603, (1568, 512), (512, 1), 0), permute_595, out=buf604)
        del permute_595
        buf605 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf603, (512, 1568), (1, 512), 0), view_301, out=buf605)
        del view_301
        buf606 = reinterpret_tensor(buf600, (1, 512, 13), (6656, 1, 512), 0); del buf600  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf603, buf606, 6656, 121, grid=grid(6656), stream=stream0)
        buf607 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf606, buf607, 512, 13, grid=grid(512), stream=stream0)
        buf608 = reinterpret_tensor(buf604, (8, 196, 2048), (401408, 2048, 1), 0); del buf604  # reuse
        # Source Nodes: [x_207], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_27.run(buf608, addmm_42, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_42
        buf609 = reinterpret_tensor(buf603, (1568, 512), (512, 1), 0); del buf603  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf608, (1568, 2048), (2048, 1), 0), permute_599, out=buf609)
        del permute_599
        buf610 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf608, (2048, 1568), (1, 2048), 0), view_299, out=buf610)
        del view_299
        buf611 = buf566; del buf566  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf608, buf611, 26624, 121, grid=grid(26624), stream=stream0)
        buf612 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf611, buf612, 2048, 13, grid=grid(2048), stream=stream0)
        buf619 = reinterpret_tensor(buf602, (8, 196, 512), (100352, 512, 1), 0); del buf602  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf619, buf609, primals_161, mul_108, div_99, 1568, 512, grid=grid(1568), stream=stream0)
        del div_99
        del primals_161
        buf615 = reinterpret_tensor(buf606, (512, 13), (1, 512), 0); del buf606  # reuse
        buf617 = reinterpret_tensor(buf598, (512, 13), (1, 512), 0); del buf598  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_31.run(buf609, mul_108, buf615, buf617, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_108
        buf616 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf615, buf616, 512, 13, grid=grid(512), stream=stream0)
        buf618 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf617, buf618, 512, 13, grid=grid(512), stream=stream0)
        buf620 = reinterpret_tensor(buf609, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf609  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_65.run(buf619, bernoulli_18, buf620, 802816, grid=grid(802816), stream=stream0)
        del bernoulli_18
        buf621 = buf592; del buf592  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf620, (1568, 512), (512, 1), 0), permute_604, out=buf621)
        del permute_604
        buf622 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf620, (512, 1568), (1, 512), 0), view_293, out=buf622)
        del view_293
        buf623 = reinterpret_tensor(buf617, (1, 512, 13), (6656, 1, 512), 0); del buf617  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf620, buf623, 6656, 121, grid=grid(6656), stream=stream0)
        buf624 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf623, buf624, 512, 13, grid=grid(512), stream=stream0)
        buf625 = reinterpret_tensor(buf620, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf620  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf621, buf625, 802816, grid=grid(802816), stream=stream0)
        buf626 = reinterpret_tensor(buf621, (512, 49, 32), (1568, 32, 1), 0); del buf621  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_609, reinterpret_tensor(buf625, (512, 49, 32), (1568, 32, 1), 0), out=buf626)
        del permute_609
        buf627 = reinterpret_tensor(buf588, (512, 49, 49), (2401, 49, 1), 0); del buf588  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf625, (512, 49, 32), (1568, 32, 1), 0), permute_610, out=buf627)
        del permute_610
        buf628 = buf583; del buf583  # reuse
        buf633 = reinterpret_tensor(buf582, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf582  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_34.run(buf627, alias_37, buf628, buf633, 25088, 49, grid=grid(25088), stream=stream0)
        buf629 = buf584; del buf584  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused_sum_35.run(buf627, alias_37, buf628, buf629, 38416, 32, grid=grid(38416), stream=stream0)
        del alias_37
        buf630 = empty((169, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_36.run(buf630, 2704, grid=grid(2704), stream=stream0)
        aten.index_put_(buf630, [view_287], reinterpret_tensor(buf629, (2401, 16), (1, 2401), 0), True)
        del view_287
        buf634 = reinterpret_tensor(buf625, (512, 32, 49), (1568, 49, 1), 0); del buf625  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_612, reinterpret_tensor(buf633, (512, 49, 49), (2401, 49, 1), 0), out=buf634)
        del permute_612
        buf635 = buf581; del buf581  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf633, (512, 49, 49), (2401, 49, 1), 0), permute_613, out=buf635)
        del permute_613
        buf636 = buf591; del buf591  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf635, buf634, buf626, buf636, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf637 = reinterpret_tensor(buf635, (1568, 512), (512, 1), 0); del buf635  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf636, (1568, 1536), (1536, 1), 0), permute_616, out=buf637)
        del permute_616
        buf638 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf636, (1536, 1568), (1, 1536), 0), view_281, out=buf638)
        del view_281
        buf639 = buf594; del buf594  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf636, buf639, 19968, 121, grid=grid(19968), stream=stream0)
        buf640 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_39.run(buf639, buf640, 1536, 13, grid=grid(1536), stream=stream0)
        buf647 = reinterpret_tensor(buf619, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf619  # reuse
        buf648 = reinterpret_tensor(buf634, (8, 196, 512), (100352, 512, 1), 0); del buf634  # reuse
        # Source Nodes: [div__17], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_66.run(buf647, buf637, primals_155, mul_104, div_100, bernoulli_17, buf648, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_17
        del div_100
        del primals_155
        buf643 = reinterpret_tensor(buf623, (512, 13), (1, 512), 0); del buf623  # reuse
        buf645 = buf615; del buf615  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_46.run(buf637, mul_104, buf643, buf645, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_104
        buf644 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf643, buf644, 512, 13, grid=grid(512), stream=stream0)
        buf646 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf645, buf646, 512, 13, grid=grid(512), stream=stream0)
        buf649 = reinterpret_tensor(buf608, (1568, 2048), (2048, 1), 0); del buf608  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf648, (1568, 512), (512, 1), 0), permute_621, out=buf649)
        del permute_621
        buf650 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf648, (512, 1568), (1, 512), 0), view_275, out=buf650)
        del view_275
        buf651 = reinterpret_tensor(buf645, (1, 512, 13), (6656, 1, 512), 0); del buf645  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf648, buf651, 6656, 121, grid=grid(6656), stream=stream0)
        buf652 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf651, buf652, 512, 13, grid=grid(512), stream=stream0)
        buf653 = reinterpret_tensor(buf649, (8, 196, 2048), (401408, 2048, 1), 0); del buf649  # reuse
        # Source Nodes: [x_189], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_27.run(buf653, addmm_38, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_38
        buf654 = reinterpret_tensor(buf648, (1568, 512), (512, 1), 0); del buf648  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf653, (1568, 2048), (2048, 1), 0), permute_625, out=buf654)
        del permute_625
        buf655 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf653, (2048, 1568), (1, 2048), 0), view_273, out=buf655)
        del view_273
        buf656 = buf611; del buf611  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf653, buf656, 26624, 121, grid=grid(26624), stream=stream0)
        buf657 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf656, buf657, 2048, 13, grid=grid(2048), stream=stream0)
        buf664 = reinterpret_tensor(buf647, (8, 196, 512), (100352, 512, 1), 0); del buf647  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf664, buf654, primals_149, mul_98, div_101, 1568, 512, grid=grid(1568), stream=stream0)
        del div_101
        del primals_149
        buf660 = reinterpret_tensor(buf651, (512, 13), (1, 512), 0); del buf651  # reuse
        buf662 = buf643; del buf643  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_31.run(buf654, mul_98, buf660, buf662, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_98
        buf661 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf660, buf661, 512, 13, grid=grid(512), stream=stream0)
        buf663 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf662, buf663, 512, 13, grid=grid(512), stream=stream0)
        buf665 = reinterpret_tensor(buf654, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf654  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_67.run(buf664, bernoulli_16, buf665, 802816, grid=grid(802816), stream=stream0)
        del bernoulli_16
        buf666 = buf637; del buf637  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf665, (1568, 512), (512, 1), 0), permute_630, out=buf666)
        del permute_630
        buf667 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf665, (512, 1568), (1, 512), 0), view_267, out=buf667)
        del view_267
        buf668 = reinterpret_tensor(buf662, (1, 512, 13), (6656, 1, 512), 0); del buf662  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf665, buf668, 6656, 121, grid=grid(6656), stream=stream0)
        buf669 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf668, buf669, 512, 13, grid=grid(512), stream=stream0)
        buf670 = reinterpret_tensor(buf665, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf665  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf666, buf670, 802816, grid=grid(802816), stream=stream0)
        buf671 = reinterpret_tensor(buf666, (512, 49, 32), (1568, 32, 1), 0); del buf666  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_635, reinterpret_tensor(buf670, (512, 49, 32), (1568, 32, 1), 0), out=buf671)
        del permute_635
        buf672 = reinterpret_tensor(buf633, (512, 49, 49), (2401, 49, 1), 0); del buf633  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf670, (512, 49, 32), (1568, 32, 1), 0), permute_636, out=buf672)
        del permute_636
        buf673 = buf628; del buf628  # reuse
        buf678 = reinterpret_tensor(buf627, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf627  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_34.run(buf672, alias_38, buf673, buf678, 25088, 49, grid=grid(25088), stream=stream0)
        buf674 = buf629; del buf629  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_35.run(buf672, alias_38, buf673, buf674, 38416, 32, grid=grid(38416), stream=stream0)
        del alias_38
        buf675 = empty((169, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_36.run(buf675, 2704, grid=grid(2704), stream=stream0)
        aten.index_put_(buf675, [view_259], reinterpret_tensor(buf674, (2401, 16), (1, 2401), 0), True)
        del view_259
        buf679 = reinterpret_tensor(buf670, (512, 32, 49), (1568, 49, 1), 0); del buf670  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_638, reinterpret_tensor(buf678, (512, 49, 49), (2401, 49, 1), 0), out=buf679)
        del permute_638
        buf680 = buf626; del buf626  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf678, (512, 49, 49), (2401, 49, 1), 0), permute_639, out=buf680)
        del permute_639
        buf681 = buf636; del buf636  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf680, buf679, buf671, buf681, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf682 = reinterpret_tensor(buf680, (1568, 512), (512, 1), 0); del buf680  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf681, (1568, 1536), (1536, 1), 0), permute_642, out=buf682)
        del permute_642
        buf683 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf681, (1536, 1568), (1, 1536), 0), view_253, out=buf683)
        del view_253
        buf684 = buf639; del buf639  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf681, buf684, 19968, 121, grid=grid(19968), stream=stream0)
        buf685 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_39.run(buf684, buf685, 1536, 13, grid=grid(1536), stream=stream0)
        buf692 = reinterpret_tensor(buf664, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf664  # reuse
        buf693 = reinterpret_tensor(buf679, (8, 196, 512), (100352, 512, 1), 0); del buf679  # reuse
        # Source Nodes: [div__15], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_add_div_mul_native_layer_norm_backward_roll_68.run(buf692, buf682, primals_143, mul_94, div_102, bernoulli_15, buf693, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_15
        del div_102
        del primals_143
        buf688 = reinterpret_tensor(buf668, (512, 13), (13, 1), 0); del buf668  # reuse
        buf690 = reinterpret_tensor(buf660, (512, 13), (13, 1), 0); del buf660  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_red_fused_native_layer_norm_backward_roll_41.run(buf682, mul_94, buf688, buf690, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_94
        buf689 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_42.run(buf688, buf689, 512, 13, grid=grid(512), stream=stream0)
        buf691 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_42.run(buf690, buf691, 512, 13, grid=grid(512), stream=stream0)
        buf694 = reinterpret_tensor(buf653, (1568, 2048), (2048, 1), 0); del buf653  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf693, (1568, 512), (512, 1), 0), permute_647, out=buf694)
        del permute_647
        buf695 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf693, (512, 1568), (1, 512), 0), view_247, out=buf695)
        del view_247
        buf696 = reinterpret_tensor(buf690, (1, 512, 13), (6656, 1, 512), 0); del buf690  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf693, buf696, 6656, 121, grid=grid(6656), stream=stream0)
        buf697 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf696, buf697, 512, 13, grid=grid(512), stream=stream0)
        buf698 = reinterpret_tensor(buf694, (8, 196, 2048), (401408, 2048, 1), 0); del buf694  # reuse
        # Source Nodes: [x_171], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_27.run(buf698, addmm_34, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_34
        buf699 = reinterpret_tensor(buf693, (1568, 512), (512, 1), 0); del buf693  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf698, (1568, 2048), (2048, 1), 0), permute_651, out=buf699)
        del permute_651
        buf700 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf698, (2048, 1568), (1, 2048), 0), view_245, out=buf700)
        del view_245
        buf701 = buf656; del buf656  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf698, buf701, 26624, 121, grid=grid(26624), stream=stream0)
        buf702 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf701, buf702, 2048, 13, grid=grid(2048), stream=stream0)
        buf709 = reinterpret_tensor(buf692, (8, 196, 512), (100352, 512, 1), 0); del buf692  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf709, buf699, primals_137, mul_88, div_103, 1568, 512, grid=grid(1568), stream=stream0)
        del div_103
        del primals_137
        buf705 = reinterpret_tensor(buf696, (512, 13), (1, 512), 0); del buf696  # reuse
        buf707 = reinterpret_tensor(buf688, (512, 13), (1, 512), 0); del buf688  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_31.run(buf699, mul_88, buf705, buf707, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_88
        buf706 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf705, buf706, 512, 13, grid=grid(512), stream=stream0)
        buf708 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf707, buf708, 512, 13, grid=grid(512), stream=stream0)
        buf710 = reinterpret_tensor(buf699, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf699  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_69.run(buf709, bernoulli_14, buf710, 802816, grid=grid(802816), stream=stream0)
        del bernoulli_14
        buf711 = buf682; del buf682  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf710, (1568, 512), (512, 1), 0), permute_656, out=buf711)
        del permute_656
        buf712 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf710, (512, 1568), (1, 512), 0), view_239, out=buf712)
        del view_239
        buf713 = reinterpret_tensor(buf707, (1, 512, 13), (6656, 1, 512), 0); del buf707  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf710, buf713, 6656, 121, grid=grid(6656), stream=stream0)
        buf714 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf713, buf714, 512, 13, grid=grid(512), stream=stream0)
        buf715 = reinterpret_tensor(buf710, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf710  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf711, buf715, 802816, grid=grid(802816), stream=stream0)
        buf716 = reinterpret_tensor(buf711, (512, 49, 32), (1568, 32, 1), 0); del buf711  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_661, reinterpret_tensor(buf715, (512, 49, 32), (1568, 32, 1), 0), out=buf716)
        del permute_661
        buf717 = reinterpret_tensor(buf678, (512, 49, 49), (2401, 49, 1), 0); del buf678  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf715, (512, 49, 32), (1568, 32, 1), 0), permute_662, out=buf717)
        del permute_662
        buf718 = buf673; del buf673  # reuse
        buf723 = reinterpret_tensor(buf672, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf672  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_34.run(buf717, alias_39, buf718, buf723, 25088, 49, grid=grid(25088), stream=stream0)
        buf719 = buf674; del buf674  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused_sum_35.run(buf717, alias_39, buf718, buf719, 38416, 32, grid=grid(38416), stream=stream0)
        del alias_39
        buf720 = empty((169, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_36.run(buf720, 2704, grid=grid(2704), stream=stream0)
        aten.index_put_(buf720, [view_233], reinterpret_tensor(buf719, (2401, 16), (1, 2401), 0), True)
        del view_233
        buf724 = reinterpret_tensor(buf715, (512, 32, 49), (1568, 49, 1), 0); del buf715  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_664, reinterpret_tensor(buf723, (512, 49, 49), (2401, 49, 1), 0), out=buf724)
        del permute_664
        buf725 = buf671; del buf671  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf723, (512, 49, 49), (2401, 49, 1), 0), permute_665, out=buf725)
        del permute_665
        buf726 = buf681; del buf681  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf725, buf724, buf716, buf726, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf727 = reinterpret_tensor(buf725, (1568, 512), (512, 1), 0); del buf725  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf726, (1568, 1536), (1536, 1), 0), permute_668, out=buf727)
        del permute_668
        buf728 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf726, (1536, 1568), (1, 1536), 0), view_227, out=buf728)
        del view_227
        buf729 = buf684; del buf684  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf726, buf729, 19968, 121, grid=grid(19968), stream=stream0)
        buf730 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_39.run(buf729, buf730, 1536, 13, grid=grid(1536), stream=stream0)
        buf737 = reinterpret_tensor(buf709, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf709  # reuse
        buf738 = reinterpret_tensor(buf724, (8, 196, 512), (100352, 512, 1), 0); del buf724  # reuse
        # Source Nodes: [div__13], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_70.run(buf737, buf727, primals_131, mul_84, div_104, bernoulli_13, buf738, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_13
        del div_104
        del primals_131
        buf733 = reinterpret_tensor(buf713, (512, 13), (1, 512), 0); del buf713  # reuse
        buf735 = buf705; del buf705  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_46.run(buf727, mul_84, buf733, buf735, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_84
        buf734 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf733, buf734, 512, 13, grid=grid(512), stream=stream0)
        buf736 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf735, buf736, 512, 13, grid=grid(512), stream=stream0)
        buf739 = reinterpret_tensor(buf698, (1568, 2048), (2048, 1), 0); del buf698  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf738, (1568, 512), (512, 1), 0), permute_673, out=buf739)
        del permute_673
        buf740 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf738, (512, 1568), (1, 512), 0), view_221, out=buf740)
        del view_221
        buf741 = reinterpret_tensor(buf735, (1, 512, 13), (6656, 1, 512), 0); del buf735  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf738, buf741, 6656, 121, grid=grid(6656), stream=stream0)
        buf742 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf741, buf742, 512, 13, grid=grid(512), stream=stream0)
        buf743 = reinterpret_tensor(buf739, (8, 196, 2048), (401408, 2048, 1), 0); del buf739  # reuse
        # Source Nodes: [x_153], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_27.run(buf743, addmm_30, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_30
        buf744 = reinterpret_tensor(buf738, (1568, 512), (512, 1), 0); del buf738  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf743, (1568, 2048), (2048, 1), 0), permute_677, out=buf744)
        del permute_677
        buf745 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf743, (2048, 1568), (1, 2048), 0), view_219, out=buf745)
        del view_219
        buf746 = buf701; del buf701  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf743, buf746, 26624, 121, grid=grid(26624), stream=stream0)
        buf747 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf746, buf747, 2048, 13, grid=grid(2048), stream=stream0)
        buf754 = reinterpret_tensor(buf737, (8, 196, 512), (100352, 512, 1), 0); del buf737  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf754, buf744, primals_125, mul_78, div_105, 1568, 512, grid=grid(1568), stream=stream0)
        del div_105
        del primals_125
        buf750 = reinterpret_tensor(buf741, (512, 13), (1, 512), 0); del buf741  # reuse
        buf752 = buf733; del buf733  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_31.run(buf744, mul_78, buf750, buf752, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_78
        buf751 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf750, buf751, 512, 13, grid=grid(512), stream=stream0)
        buf753 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf752, buf753, 512, 13, grid=grid(512), stream=stream0)
        buf755 = reinterpret_tensor(buf744, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf744  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_71.run(buf754, bernoulli_12, buf755, 802816, grid=grid(802816), stream=stream0)
        del bernoulli_12
        buf756 = buf727; del buf727  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf755, (1568, 512), (512, 1), 0), permute_682, out=buf756)
        del permute_682
        buf757 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf755, (512, 1568), (1, 512), 0), view_213, out=buf757)
        del view_213
        buf758 = reinterpret_tensor(buf752, (1, 512, 13), (6656, 1, 512), 0); del buf752  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf755, buf758, 6656, 121, grid=grid(6656), stream=stream0)
        buf759 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf758, buf759, 512, 13, grid=grid(512), stream=stream0)
        buf760 = reinterpret_tensor(buf755, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf755  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf756, buf760, 802816, grid=grid(802816), stream=stream0)
        buf761 = reinterpret_tensor(buf756, (512, 49, 32), (1568, 32, 1), 0); del buf756  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_687, reinterpret_tensor(buf760, (512, 49, 32), (1568, 32, 1), 0), out=buf761)
        del permute_687
        buf762 = reinterpret_tensor(buf723, (512, 49, 49), (2401, 49, 1), 0); del buf723  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf760, (512, 49, 32), (1568, 32, 1), 0), permute_688, out=buf762)
        del permute_688
        buf763 = buf718; del buf718  # reuse
        buf768 = reinterpret_tensor(buf717, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf717  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_34.run(buf762, alias_40, buf763, buf768, 25088, 49, grid=grid(25088), stream=stream0)
        buf764 = buf719; del buf719  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_35.run(buf762, alias_40, buf763, buf764, 38416, 32, grid=grid(38416), stream=stream0)
        del alias_40
        buf765 = empty((169, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_36.run(buf765, 2704, grid=grid(2704), stream=stream0)
        aten.index_put_(buf765, [view_205], reinterpret_tensor(buf764, (2401, 16), (1, 2401), 0), True)
        del view_205
        buf769 = reinterpret_tensor(buf760, (512, 32, 49), (1568, 49, 1), 0); del buf760  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_690, reinterpret_tensor(buf768, (512, 49, 49), (2401, 49, 1), 0), out=buf769)
        del permute_690
        buf770 = buf716; del buf716  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf768, (512, 49, 49), (2401, 49, 1), 0), permute_691, out=buf770)
        del permute_691
        buf771 = buf726; del buf726  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf770, buf769, buf761, buf771, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf772 = reinterpret_tensor(buf770, (1568, 512), (512, 1), 0); del buf770  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf771, (1568, 1536), (1536, 1), 0), permute_694, out=buf772)
        del permute_694
        buf773 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf771, (1536, 1568), (1, 1536), 0), view_199, out=buf773)
        del view_199
        buf774 = buf729; del buf729  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf771, buf774, 19968, 121, grid=grid(19968), stream=stream0)
        buf775 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_39.run(buf774, buf775, 1536, 13, grid=grid(1536), stream=stream0)
        buf782 = reinterpret_tensor(buf754, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf754  # reuse
        buf783 = reinterpret_tensor(buf769, (8, 196, 512), (100352, 512, 1), 0); del buf769  # reuse
        # Source Nodes: [div__11], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_add_div_mul_native_layer_norm_backward_roll_72.run(buf782, buf772, primals_119, mul_74, div_106, bernoulli_11, buf783, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_11
        del div_106
        del primals_119
        buf778 = reinterpret_tensor(buf758, (512, 13), (13, 1), 0); del buf758  # reuse
        buf780 = reinterpret_tensor(buf750, (512, 13), (13, 1), 0); del buf750  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_red_fused_native_layer_norm_backward_roll_41.run(buf772, mul_74, buf778, buf780, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_74
        buf779 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_42.run(buf778, buf779, 512, 13, grid=grid(512), stream=stream0)
        buf781 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_42.run(buf780, buf781, 512, 13, grid=grid(512), stream=stream0)
        buf784 = reinterpret_tensor(buf743, (1568, 2048), (2048, 1), 0); del buf743  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf783, (1568, 512), (512, 1), 0), permute_699, out=buf784)
        del permute_699
        buf785 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf783, (512, 1568), (1, 512), 0), view_193, out=buf785)
        del view_193
        buf786 = reinterpret_tensor(buf780, (1, 512, 13), (6656, 1, 512), 0); del buf780  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf783, buf786, 6656, 121, grid=grid(6656), stream=stream0)
        buf787 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf786, buf787, 512, 13, grid=grid(512), stream=stream0)
        buf788 = reinterpret_tensor(buf784, (8, 196, 2048), (401408, 2048, 1), 0); del buf784  # reuse
        # Source Nodes: [x_135], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_27.run(buf788, addmm_26, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_26
        buf789 = reinterpret_tensor(buf783, (1568, 512), (512, 1), 0); del buf783  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf788, (1568, 2048), (2048, 1), 0), permute_703, out=buf789)
        del permute_703
        buf790 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf788, (2048, 1568), (1, 2048), 0), view_191, out=buf790)
        del view_191
        buf791 = buf746; del buf746  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf788, buf791, 26624, 121, grid=grid(26624), stream=stream0)
        buf792 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf791, buf792, 2048, 13, grid=grid(2048), stream=stream0)
        buf799 = reinterpret_tensor(buf782, (8, 196, 512), (100352, 512, 1), 0); del buf782  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf799, buf789, primals_113, mul_68, div_107, 1568, 512, grid=grid(1568), stream=stream0)
        del div_107
        del primals_113
        buf795 = reinterpret_tensor(buf786, (512, 13), (1, 512), 0); del buf786  # reuse
        buf797 = reinterpret_tensor(buf778, (512, 13), (1, 512), 0); del buf778  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_31.run(buf789, mul_68, buf795, buf797, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_68
        buf796 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf795, buf796, 512, 13, grid=grid(512), stream=stream0)
        buf798 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf797, buf798, 512, 13, grid=grid(512), stream=stream0)
        buf800 = reinterpret_tensor(buf789, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf789  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_73.run(buf799, bernoulli_10, buf800, 802816, grid=grid(802816), stream=stream0)
        del bernoulli_10
        buf801 = buf772; del buf772  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf800, (1568, 512), (512, 1), 0), permute_708, out=buf801)
        del permute_708
        buf802 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf800, (512, 1568), (1, 512), 0), view_185, out=buf802)
        del view_185
        buf803 = reinterpret_tensor(buf797, (1, 512, 13), (6656, 1, 512), 0); del buf797  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf800, buf803, 6656, 121, grid=grid(6656), stream=stream0)
        buf804 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf803, buf804, 512, 13, grid=grid(512), stream=stream0)
        buf805 = reinterpret_tensor(buf800, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf800  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf801, buf805, 802816, grid=grid(802816), stream=stream0)
        buf806 = reinterpret_tensor(buf801, (512, 49, 32), (1568, 32, 1), 0); del buf801  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_713, reinterpret_tensor(buf805, (512, 49, 32), (1568, 32, 1), 0), out=buf806)
        del permute_713
        buf807 = reinterpret_tensor(buf768, (512, 49, 49), (2401, 49, 1), 0); del buf768  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf805, (512, 49, 32), (1568, 32, 1), 0), permute_714, out=buf807)
        del permute_714
        buf808 = buf763; del buf763  # reuse
        buf813 = reinterpret_tensor(buf762, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf762  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_34.run(buf807, alias_41, buf808, buf813, 25088, 49, grid=grid(25088), stream=stream0)
        buf809 = buf764; del buf764  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused_sum_35.run(buf807, alias_41, buf808, buf809, 38416, 32, grid=grid(38416), stream=stream0)
        del alias_41
        buf810 = empty((169, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_36.run(buf810, 2704, grid=grid(2704), stream=stream0)
        aten.index_put_(buf810, [view_179], reinterpret_tensor(buf809, (2401, 16), (1, 2401), 0), True)
        del view_179
        buf814 = reinterpret_tensor(buf805, (512, 32, 49), (1568, 49, 1), 0); del buf805  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_716, reinterpret_tensor(buf813, (512, 49, 49), (2401, 49, 1), 0), out=buf814)
        del permute_716
        buf815 = buf761; del buf761  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf813, (512, 49, 49), (2401, 49, 1), 0), permute_717, out=buf815)
        del permute_717
        buf816 = buf771; del buf771  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf815, buf814, buf806, buf816, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf817 = reinterpret_tensor(buf815, (1568, 512), (512, 1), 0); del buf815  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf816, (1568, 1536), (1536, 1), 0), permute_720, out=buf817)
        del permute_720
        buf818 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf816, (1536, 1568), (1, 1536), 0), view_173, out=buf818)
        del view_173
        buf819 = buf774; del buf774  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf816, buf819, 19968, 121, grid=grid(19968), stream=stream0)
        buf820 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_39.run(buf819, buf820, 1536, 13, grid=grid(1536), stream=stream0)
        buf827 = reinterpret_tensor(buf799, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf799  # reuse
        buf828 = reinterpret_tensor(buf814, (8, 196, 512), (100352, 512, 1), 0); del buf814  # reuse
        # Source Nodes: [div__9], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_74.run(buf827, buf817, primals_107, mul_64, div_108, bernoulli_9, buf828, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_9
        del div_108
        del primals_107
        buf823 = reinterpret_tensor(buf803, (512, 13), (1, 512), 0); del buf803  # reuse
        buf825 = buf795; del buf795  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_46.run(buf817, mul_64, buf823, buf825, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_64
        buf824 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf823, buf824, 512, 13, grid=grid(512), stream=stream0)
        buf826 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf825, buf826, 512, 13, grid=grid(512), stream=stream0)
        buf829 = reinterpret_tensor(buf788, (1568, 2048), (2048, 1), 0); del buf788  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf828, (1568, 512), (512, 1), 0), permute_725, out=buf829)
        del permute_725
        buf830 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf828, (512, 1568), (1, 512), 0), view_167, out=buf830)
        del view_167
        buf831 = reinterpret_tensor(buf825, (1, 512, 13), (6656, 1, 512), 0); del buf825  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf828, buf831, 6656, 121, grid=grid(6656), stream=stream0)
        buf832 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf831, buf832, 512, 13, grid=grid(512), stream=stream0)
        buf833 = reinterpret_tensor(buf829, (8, 196, 2048), (401408, 2048, 1), 0); del buf829  # reuse
        # Source Nodes: [x_117], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_27.run(buf833, addmm_22, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_22
        buf834 = reinterpret_tensor(buf828, (1568, 512), (512, 1), 0); del buf828  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf833, (1568, 2048), (2048, 1), 0), permute_729, out=buf834)
        del permute_729
        buf835 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf833, (2048, 1568), (1, 2048), 0), view_165, out=buf835)
        del view_165
        buf836 = buf791; del buf791  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf833, buf836, 26624, 121, grid=grid(26624), stream=stream0)
        buf837 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf836, buf837, 2048, 13, grid=grid(2048), stream=stream0)
        buf844 = reinterpret_tensor(buf827, (8, 196, 512), (100352, 512, 1), 0); del buf827  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf844, buf834, primals_101, mul_58, div_109, 1568, 512, grid=grid(1568), stream=stream0)
        del div_109
        del primals_101
        buf840 = reinterpret_tensor(buf831, (512, 13), (1, 512), 0); del buf831  # reuse
        buf842 = buf823; del buf823  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_31.run(buf834, mul_58, buf840, buf842, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_58
        buf841 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf840, buf841, 512, 13, grid=grid(512), stream=stream0)
        buf843 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf842, buf843, 512, 13, grid=grid(512), stream=stream0)
        buf845 = reinterpret_tensor(buf834, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf834  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_75.run(buf844, bernoulli_8, buf845, 802816, grid=grid(802816), stream=stream0)
        del bernoulli_8
        buf846 = buf817; del buf817  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf845, (1568, 512), (512, 1), 0), permute_734, out=buf846)
        del permute_734
        buf847 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf845, (512, 1568), (1, 512), 0), view_159, out=buf847)
        del view_159
        buf848 = reinterpret_tensor(buf842, (1, 512, 13), (6656, 1, 512), 0); del buf842  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf845, buf848, 6656, 121, grid=grid(6656), stream=stream0)
        buf849 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf848, buf849, 512, 13, grid=grid(512), stream=stream0)
        buf850 = reinterpret_tensor(buf845, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf845  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf846, buf850, 802816, grid=grid(802816), stream=stream0)
        buf851 = reinterpret_tensor(buf846, (512, 49, 32), (1568, 32, 1), 0); del buf846  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_739, reinterpret_tensor(buf850, (512, 49, 32), (1568, 32, 1), 0), out=buf851)
        del permute_739
        buf852 = reinterpret_tensor(buf813, (512, 49, 49), (2401, 49, 1), 0); del buf813  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf850, (512, 49, 32), (1568, 32, 1), 0), permute_740, out=buf852)
        del permute_740
        buf853 = buf808; del buf808  # reuse
        buf858 = reinterpret_tensor(buf807, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf807  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_34.run(buf852, alias_42, buf853, buf858, 25088, 49, grid=grid(25088), stream=stream0)
        buf854 = buf809; del buf809  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_35.run(buf852, alias_42, buf853, buf854, 38416, 32, grid=grid(38416), stream=stream0)
        del alias_42
        buf855 = empty((169, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_36.run(buf855, 2704, grid=grid(2704), stream=stream0)
        aten.index_put_(buf855, [view_151], reinterpret_tensor(buf854, (2401, 16), (1, 2401), 0), True)
        del view_151
        buf859 = reinterpret_tensor(buf850, (512, 32, 49), (1568, 49, 1), 0); del buf850  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_742, reinterpret_tensor(buf858, (512, 49, 49), (2401, 49, 1), 0), out=buf859)
        del permute_742
        buf860 = buf806; del buf806  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf858, (512, 49, 49), (2401, 49, 1), 0), permute_743, out=buf860)
        del permute_743
        buf861 = buf816; del buf816  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf860, buf859, buf851, buf861, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf862 = reinterpret_tensor(buf860, (1568, 512), (512, 1), 0); del buf860  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf861, (1568, 1536), (1536, 1), 0), permute_746, out=buf862)
        del permute_746
        buf863 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf861, (1536, 1568), (1, 1536), 0), view_145, out=buf863)
        del view_145
        buf864 = buf819; del buf819  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf861, buf864, 19968, 121, grid=grid(19968), stream=stream0)
        buf865 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_39.run(buf864, buf865, 1536, 13, grid=grid(1536), stream=stream0)
        buf872 = reinterpret_tensor(buf844, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf844  # reuse
        buf873 = reinterpret_tensor(buf859, (8, 196, 512), (100352, 512, 1), 0); del buf859  # reuse
        # Source Nodes: [div__7], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_add_div_mul_native_layer_norm_backward_roll_76.run(buf872, buf862, primals_95, mul_54, div_110, bernoulli_7, buf873, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_7
        del div_110
        del primals_95
        buf868 = reinterpret_tensor(buf848, (512, 13), (13, 1), 0); del buf848  # reuse
        buf870 = reinterpret_tensor(buf840, (512, 13), (13, 1), 0); del buf840  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_red_fused_native_layer_norm_backward_roll_41.run(buf862, mul_54, buf868, buf870, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_54
        buf869 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_42.run(buf868, buf869, 512, 13, grid=grid(512), stream=stream0)
        buf871 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_42.run(buf870, buf871, 512, 13, grid=grid(512), stream=stream0)
        buf874 = reinterpret_tensor(buf833, (1568, 2048), (2048, 1), 0); del buf833  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf873, (1568, 512), (512, 1), 0), permute_751, out=buf874)
        del permute_751
        buf875 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf873, (512, 1568), (1, 512), 0), view_139, out=buf875)
        del view_139
        buf876 = reinterpret_tensor(buf870, (1, 512, 13), (6656, 1, 512), 0); del buf870  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf873, buf876, 6656, 121, grid=grid(6656), stream=stream0)
        buf877 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf876, buf877, 512, 13, grid=grid(512), stream=stream0)
        buf878 = reinterpret_tensor(buf874, (8, 196, 2048), (401408, 2048, 1), 0); del buf874  # reuse
        # Source Nodes: [x_99], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_27.run(buf878, addmm_18, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_18
        buf879 = reinterpret_tensor(buf873, (1568, 512), (512, 1), 0); del buf873  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf878, (1568, 2048), (2048, 1), 0), permute_755, out=buf879)
        del permute_755
        buf880 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf878, (2048, 1568), (1, 2048), 0), view_137, out=buf880)
        del view_137
        buf881 = buf836; del buf836  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf878, buf881, 26624, 121, grid=grid(26624), stream=stream0)
        buf882 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf881, buf882, 2048, 13, grid=grid(2048), stream=stream0)
        del buf881
        buf889 = reinterpret_tensor(buf872, (8, 196, 512), (100352, 512, 1), 0); del buf872  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf889, buf879, primals_89, mul_48, div_111, 1568, 512, grid=grid(1568), stream=stream0)
        del div_111
        del primals_89
        buf885 = reinterpret_tensor(buf876, (512, 13), (1, 512), 0); del buf876  # reuse
        buf887 = reinterpret_tensor(buf868, (512, 13), (1, 512), 0); del buf868  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_31.run(buf879, mul_48, buf885, buf887, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_48
        buf886 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf885, buf886, 512, 13, grid=grid(512), stream=stream0)
        buf888 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf887, buf888, 512, 13, grid=grid(512), stream=stream0)
        buf890 = reinterpret_tensor(buf879, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf879  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_77.run(buf889, bernoulli_6, buf890, 802816, grid=grid(802816), stream=stream0)
        del bernoulli_6
        buf891 = buf862; del buf862  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf890, (1568, 512), (512, 1), 0), permute_760, out=buf891)
        del permute_760
        buf892 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf890, (512, 1568), (1, 512), 0), view_131, out=buf892)
        del view_131
        buf893 = reinterpret_tensor(buf887, (1, 512, 13), (6656, 1, 512), 0); del buf887  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_25.run(buf890, buf893, 6656, 121, grid=grid(6656), stream=stream0)
        buf894 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf893, buf894, 512, 13, grid=grid(512), stream=stream0)
        buf895 = reinterpret_tensor(buf890, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf890  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf891, buf895, 802816, grid=grid(802816), stream=stream0)
        buf896 = reinterpret_tensor(buf891, (512, 49, 32), (1568, 32, 1), 0); del buf891  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_765, reinterpret_tensor(buf895, (512, 49, 32), (1568, 32, 1), 0), out=buf896)
        del permute_765
        buf897 = reinterpret_tensor(buf858, (512, 49, 49), (2401, 49, 1), 0); del buf858  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf895, (512, 49, 32), (1568, 32, 1), 0), permute_766, out=buf897)
        del permute_766
        buf898 = buf853; del buf853  # reuse
        buf903 = reinterpret_tensor(buf852, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf852  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_34.run(buf897, alias_43, buf898, buf903, 25088, 49, grid=grid(25088), stream=stream0)
        buf899 = buf854; del buf854  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused_sum_35.run(buf897, alias_43, buf898, buf899, 38416, 32, grid=grid(38416), stream=stream0)
        del alias_43
        del buf897
        buf900 = empty((169, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_36.run(buf900, 2704, grid=grid(2704), stream=stream0)
        aten.index_put_(buf900, [view_125], reinterpret_tensor(buf899, (2401, 16), (1, 2401), 0), True)
        del buf899
        del view_125
        buf904 = reinterpret_tensor(buf895, (512, 32, 49), (1568, 49, 1), 0); del buf895  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_768, reinterpret_tensor(buf903, (512, 49, 49), (2401, 49, 1), 0), out=buf904)
        del permute_768
        buf905 = buf851; del buf851  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf903, (512, 49, 49), (2401, 49, 1), 0), permute_769, out=buf905)
        del buf903
        del permute_769
        buf906 = buf861; del buf861  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf905, buf904, buf896, buf906, 75264, 32, grid=grid(75264, 32), stream=stream0)
        del buf896
        del buf904
        buf907 = reinterpret_tensor(buf905, (1568, 512), (512, 1), 0); del buf905  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf906, (1568, 1536), (1536, 1), 0), permute_772, out=buf907)
        del permute_772
        buf908 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf906, (1536, 1568), (1, 1536), 0), view_119, out=buf908)
        del view_119
        buf909 = buf864; del buf864  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf906, buf909, 19968, 121, grid=grid(19968), stream=stream0)
        del buf906
        buf910 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_39.run(buf909, buf910, 1536, 13, grid=grid(1536), stream=stream0)
        del buf909
        buf917 = reinterpret_tensor(buf889, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf889  # reuse
        # Source Nodes: [shifted_x_16], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_78.run(buf917, buf907, primals_83, mm_1, getitem_35, rsqrt_11, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_83
        buf913 = reinterpret_tensor(buf893, (512, 13), (1, 512), 0); del buf893  # reuse
        buf915 = buf885; del buf885  # reuse
        # Source Nodes: [shifted_x_16], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_79.run(buf907, mm_1, getitem_35, rsqrt_11, buf913, buf915, 6656, 121, grid=grid(6656), stream=stream0)
        del buf907
        del getitem_35
        del mm_1
        del rsqrt_11
        buf914 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shifted_x_16], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf913, buf914, 512, 13, grid=grid(512), stream=stream0)
        del buf913
        buf916 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_26.run(buf915, buf916, 512, 13, grid=grid(512), stream=stream0)
        del buf915
        buf918 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf917, (512, 1568), (1, 512), 0), view_114, out=buf918)
        del view_114
        buf919 = reinterpret_tensor(buf59, (1568, 1024), (1024, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf917, (1568, 512), (512, 1), 0), permute_779, out=buf919)
        del buf917
        del permute_779
        buf926 = empty_strided((8, 14, 2, 14, 2, 256), (200704, 14336, 256, 1024, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone, aten.native_layer_norm_backward]
        triton_per_fused_clone_native_layer_norm_backward_80.run(buf919, primals_80, mul_42, div_113, buf926, 1568, 1024, grid=grid(1568), stream=stream0)
        del div_113
        del primals_80
        buf922 = empty_strided((1024, 13), (1, 1024), device='cuda', dtype=torch.float32)
        buf924 = empty_strided((1024, 13), (1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_81.run(buf919, mul_42, buf922, buf924, 13312, 121, grid=grid(13312), stream=stream0)
        del mul_42
        buf923 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_82.run(buf922, buf923, 1024, 13, grid=grid(1024), stream=stream0)
        del buf922
        buf925 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_82.run(buf924, buf925, 1024, 13, grid=grid(1024), stream=stream0)
        del buf924
        buf927 = reinterpret_tensor(buf919, (8, 784, 256), (200704, 256, 1), 0); del buf919  # reuse
        # Source Nodes: [div__5], Original ATen: [aten.div, aten.mul]
        triton_poi_fused_div_mul_83.run(buf926, bernoulli_5, buf927, 1605632, grid=grid(1605632), stream=stream0)
        del bernoulli_5
        buf928 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf927, (6272, 256), (256, 1), 0), permute_782, out=buf928)
        del permute_782
        buf929 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf927, (256, 6272), (1, 256), 0), view_109, out=buf929)
        del view_109
        buf930 = reinterpret_tensor(buf79, (1, 256, 49), (12544, 1, 256), 0); del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_84.run(buf927, buf930, 12544, 128, grid=grid(12544), stream=stream0)
        buf931 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_85.run(buf930, buf931, 256, 49, grid=grid(256), stream=stream0)
        buf932 = reinterpret_tensor(buf928, (8, 784, 1024), (802816, 1024, 1), 0); del buf928  # reuse
        # Source Nodes: [x_76], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_86.run(buf932, addmm_14, 6422528, grid=grid(6422528), stream=stream0)
        del addmm_14
        buf933 = reinterpret_tensor(buf927, (6272, 256), (256, 1), 0); del buf927  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf932, (6272, 1024), (1024, 1), 0), permute_786, out=buf933)
        del permute_786
        buf934 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf932, (1024, 6272), (1, 1024), 0), view_107, out=buf934)
        del view_107
        buf935 = empty_strided((1, 1024, 49), (50176, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_87.run(buf932, buf935, 50176, 128, grid=grid(50176), stream=stream0)
        buf936 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_88.run(buf935, buf936, 1024, 49, grid=grid(1024), stream=stream0)
        buf943 = empty((8, 784, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_89.run(buf933, primals_74, mul_36, buf926, div_114, buf943, 6272, 256, grid=grid(6272), stream=stream0)
        del div_114
        del primals_74
        buf939 = reinterpret_tensor(buf930, (256, 49), (1, 256), 0); del buf930  # reuse
        buf941 = empty_strided((256, 49), (1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_90.run(buf933, mul_36, buf939, buf941, 12544, 128, grid=grid(12544), stream=stream0)
        del mul_36
        buf940 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_85.run(buf939, buf940, 256, 49, grid=grid(256), stream=stream0)
        buf942 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_85.run(buf941, buf942, 256, 49, grid=grid(256), stream=stream0)
        buf944 = reinterpret_tensor(buf933, (8, 4, 4, 7, 7, 256), (200704, 50176, 12544, 1792, 256, 1), 0); del buf933  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_91.run(buf943, bernoulli_4, buf944, 1605632, grid=grid(1605632), stream=stream0)
        del bernoulli_4
        buf945 = reinterpret_tensor(buf926, (6272, 256), (256, 1), 0); del buf926  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf944, (6272, 256), (256, 1), 0), permute_791, out=buf945)
        del permute_791
        buf946 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf944, (256, 6272), (1, 256), 0), view_101, out=buf946)
        del view_101
        buf947 = reinterpret_tensor(buf941, (1, 256, 49), (12544, 1, 256), 0); del buf941  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_84.run(buf944, buf947, 12544, 128, grid=grid(12544), stream=stream0)
        buf948 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_85.run(buf947, buf948, 256, 49, grid=grid(256), stream=stream0)
        buf949 = reinterpret_tensor(buf944, (128, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf944  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_92.run(buf945, buf949, 1605632, grid=grid(1605632), stream=stream0)
        buf950 = reinterpret_tensor(buf945, (1024, 49, 32), (1568, 32, 1), 0); del buf945  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_796, reinterpret_tensor(buf949, (1024, 49, 32), (1568, 32, 1), 0), out=buf950)
        del permute_796
        buf951 = empty((1024, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf949, (1024, 49, 32), (1568, 32, 1), 0), permute_797, out=buf951)
        del permute_797
        buf952 = reinterpret_tensor(buf935, (128, 8, 49, 1), (392, 49, 1, 50176), 0); del buf935  # reuse
        buf957 = empty((128, 8, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_93.run(buf951, alias_44, buf952, buf957, 50176, 49, grid=grid(50176), stream=stream0)
        buf953 = empty((1, 8, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_94.run(buf951, alias_44, buf952, buf953, 19208, 128, grid=grid(19208), stream=stream0)
        del alias_44
        buf954 = empty((169, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_95.run(buf954, 1352, grid=grid(1352), stream=stream0)
        aten.index_put_(buf954, [view_93], reinterpret_tensor(buf953, (2401, 8), (1, 2401), 0), True)
        del view_93
        buf958 = reinterpret_tensor(buf949, (1024, 32, 49), (1568, 49, 1), 0); del buf949  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_799, reinterpret_tensor(buf957, (1024, 49, 49), (2401, 49, 1), 0), out=buf958)
        del permute_799
        buf959 = empty((1024, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf957, (1024, 49, 49), (2401, 49, 1), 0), permute_800, out=buf959)
        del permute_800
        buf960 = empty((128, 49, 3, 8, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_96.run(buf959, buf958, buf950, buf960, 150528, 32, grid=grid(150528, 32), stream=stream0)
        buf961 = reinterpret_tensor(buf959, (6272, 256), (256, 1), 0); del buf959  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf960, (6272, 768), (768, 1), 0), permute_803, out=buf961)
        del permute_803
        buf962 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf960, (768, 6272), (1, 768), 0), view_87, out=buf962)
        del view_87
        buf963 = empty_strided((1, 768, 49), (37632, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_97.run(buf960, buf963, 37632, 128, grid=grid(37632), stream=stream0)
        buf964 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_98.run(buf963, buf964, 768, 49, grid=grid(768), stream=stream0)
        buf971 = reinterpret_tensor(buf943, (8, 28, 28, 256), (200704, 7168, 256, 1), 0); del buf943  # reuse
        buf972 = reinterpret_tensor(buf958, (8, 784, 256), (200704, 256, 1), 0); del buf958  # reuse
        # Source Nodes: [div__3], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_add_div_mul_native_layer_norm_backward_roll_99.run(buf971, buf961, primals_68, mul_32, div_115, bernoulli_3, buf972, 6272, 256, grid=grid(6272), stream=stream0)
        del bernoulli_3
        del div_115
        del primals_68
        buf967 = reinterpret_tensor(buf947, (256, 49), (49, 1), 0); del buf947  # reuse
        buf969 = reinterpret_tensor(buf939, (256, 49), (49, 1), 0); del buf939  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_red_fused_native_layer_norm_backward_roll_100.run(buf961, mul_32, buf967, buf969, 12544, 128, grid=grid(12544), stream=stream0)
        del mul_32
        buf968 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_101.run(buf967, buf968, 256, 49, grid=grid(256), stream=stream0)
        buf970 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_101.run(buf969, buf970, 256, 49, grid=grid(256), stream=stream0)
        buf973 = reinterpret_tensor(buf932, (6272, 1024), (1024, 1), 0); del buf932  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf972, (6272, 256), (256, 1), 0), permute_808, out=buf973)
        del permute_808
        buf974 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf972, (256, 6272), (1, 256), 0), view_81, out=buf974)
        del view_81
        buf975 = reinterpret_tensor(buf969, (1, 256, 49), (12544, 1, 256), 0); del buf969  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_84.run(buf972, buf975, 12544, 128, grid=grid(12544), stream=stream0)
        buf976 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_85.run(buf975, buf976, 256, 49, grid=grid(256), stream=stream0)
        buf977 = reinterpret_tensor(buf973, (8, 784, 1024), (802816, 1024, 1), 0); del buf973  # reuse
        # Source Nodes: [x_58], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_86.run(buf977, addmm_10, 6422528, grid=grid(6422528), stream=stream0)
        del addmm_10
        buf978 = reinterpret_tensor(buf972, (6272, 256), (256, 1), 0); del buf972  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf977, (6272, 1024), (1024, 1), 0), permute_812, out=buf978)
        del permute_812
        buf979 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf977, (1024, 6272), (1, 1024), 0), view_79, out=buf979)
        del view_79
        buf980 = reinterpret_tensor(buf952, (1, 1024, 49), (50176, 1, 1024), 0); del buf952  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_87.run(buf977, buf980, 50176, 128, grid=grid(50176), stream=stream0)
        del buf977
        buf981 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_88.run(buf980, buf981, 1024, 49, grid=grid(1024), stream=stream0)
        buf988 = reinterpret_tensor(buf971, (8, 784, 256), (200704, 256, 1), 0); del buf971  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_102.run(buf988, buf978, primals_62, mul_26, div_116, 6272, 256, grid=grid(6272), stream=stream0)
        del div_116
        del primals_62
        buf984 = reinterpret_tensor(buf975, (256, 49), (1, 256), 0); del buf975  # reuse
        buf986 = reinterpret_tensor(buf967, (256, 49), (1, 256), 0); del buf967  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_90.run(buf978, mul_26, buf984, buf986, 12544, 128, grid=grid(12544), stream=stream0)
        del mul_26
        buf985 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_85.run(buf984, buf985, 256, 49, grid=grid(256), stream=stream0)
        buf987 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_85.run(buf986, buf987, 256, 49, grid=grid(256), stream=stream0)
        buf989 = reinterpret_tensor(buf978, (8, 4, 4, 7, 7, 256), (200704, 50176, 12544, 1792, 256, 1), 0); del buf978  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_103.run(buf988, bernoulli_2, buf989, 1605632, grid=grid(1605632), stream=stream0)
        del bernoulli_2
        buf990 = buf961; del buf961  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf989, (6272, 256), (256, 1), 0), permute_817, out=buf990)
        del permute_817
        buf991 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf989, (256, 6272), (1, 256), 0), view_73, out=buf991)
        del view_73
        buf992 = reinterpret_tensor(buf986, (1, 256, 49), (12544, 1, 256), 0); del buf986  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_84.run(buf989, buf992, 12544, 128, grid=grid(12544), stream=stream0)
        buf993 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_85.run(buf992, buf993, 256, 49, grid=grid(256), stream=stream0)
        buf994 = reinterpret_tensor(buf989, (128, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf989  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_92.run(buf990, buf994, 1605632, grid=grid(1605632), stream=stream0)
        buf995 = reinterpret_tensor(buf990, (1024, 49, 32), (1568, 32, 1), 0); del buf990  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_822, reinterpret_tensor(buf994, (1024, 49, 32), (1568, 32, 1), 0), out=buf995)
        del permute_822
        buf996 = reinterpret_tensor(buf957, (1024, 49, 49), (2401, 49, 1), 0); del buf957  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf994, (1024, 49, 32), (1568, 32, 1), 0), permute_823, out=buf996)
        del permute_823
        buf997 = reinterpret_tensor(buf980, (128, 8, 49, 1), (392, 49, 1, 50176), 0); del buf980  # reuse
        buf1002 = reinterpret_tensor(buf951, (128, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf951  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_93.run(buf996, alias_45, buf997, buf1002, 50176, 49, grid=grid(50176), stream=stream0)
        buf998 = buf953; del buf953  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_red_fused_sum_94.run(buf996, alias_45, buf997, buf998, 19208, 128, grid=grid(19208), stream=stream0)
        del alias_45
        del buf996
        del buf997
        buf999 = empty((169, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_95.run(buf999, 1352, grid=grid(1352), stream=stream0)
        aten.index_put_(buf999, [view_67], reinterpret_tensor(buf998, (2401, 8), (1, 2401), 0), True)
        del buf998
        del view_67
        buf1003 = reinterpret_tensor(buf994, (1024, 32, 49), (1568, 49, 1), 0); del buf994  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_825, reinterpret_tensor(buf1002, (1024, 49, 49), (2401, 49, 1), 0), out=buf1003)
        del permute_825
        buf1004 = buf950; del buf950  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1002, (1024, 49, 49), (2401, 49, 1), 0), permute_826, out=buf1004)
        del buf1002
        del permute_826
        buf1005 = buf960; del buf960  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_96.run(buf1004, buf1003, buf995, buf1005, 150528, 32, grid=grid(150528, 32), stream=stream0)
        del buf1003
        del buf1004
        buf1006 = reinterpret_tensor(buf995, (6272, 256), (256, 1), 0); del buf995  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1005, (6272, 768), (768, 1), 0), permute_829, out=buf1006)
        del permute_829
        buf1007 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1005, (768, 6272), (1, 768), 0), view_61, out=buf1007)
        del view_61
        buf1008 = buf963; del buf963  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_97.run(buf1005, buf1008, 37632, 128, grid=grid(37632), stream=stream0)
        del buf1005
        buf1009 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_98.run(buf1008, buf1009, 768, 49, grid=grid(768), stream=stream0)
        del buf1008
        buf1016 = reinterpret_tensor(buf988, (8, 28, 28, 256), (200704, 7168, 256, 1), 0); del buf988  # reuse
        # Source Nodes: [shifted_x_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_104.run(buf1016, buf1006, primals_56, mm, getitem_19, rsqrt_6, 6272, 256, grid=grid(6272), stream=stream0)
        del primals_56
        buf1012 = reinterpret_tensor(buf992, (256, 49), (1, 256), 0); del buf992  # reuse
        buf1014 = buf984; del buf984  # reuse
        # Source Nodes: [shifted_x_8], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_105.run(buf1006, mm, getitem_19, rsqrt_6, buf1012, buf1014, 12544, 128, grid=grid(12544), stream=stream0)
        del buf1006
        del getitem_19
        del mm
        del rsqrt_6
        buf1013 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shifted_x_8], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_sum_85.run(buf1012, buf1013, 256, 49, grid=grid(256), stream=stream0)
        del buf1012
        buf1015 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_85.run(buf1014, buf1015, 256, 49, grid=grid(256), stream=stream0)
        del buf1014
        buf1017 = empty((256, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1016, (256, 6272), (1, 256), 0), view_56, out=buf1017)
        del view_56
        buf1018 = reinterpret_tensor(buf878, (6272, 512), (512, 1), 0); del buf878  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1016, (6272, 256), (256, 1), 0), permute_836, out=buf1018)
        del buf1016
        del permute_836
        buf1025 = empty_strided((8, 28, 2, 28, 2, 128), (401408, 14336, 128, 512, 256, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone, aten.native_layer_norm_backward]
        triton_per_fused_clone_native_layer_norm_backward_106.run(buf1018, primals_53, mul_20, div_118, buf1025, 6272, 512, grid=grid(6272), stream=stream0)
        del div_118
        del primals_53
        buf1021 = reinterpret_tensor(buf898, (512, 49), (1, 512), 0); del buf898  # reuse
        buf1023 = empty_strided((512, 49), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_107.run(buf1018, mul_20, buf1021, buf1023, 25088, 128, grid=grid(25088), stream=stream0)
        del mul_20
        buf1022 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_108.run(buf1021, buf1022, 512, 49, grid=grid(512), stream=stream0)
        buf1024 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_108.run(buf1023, buf1024, 512, 49, grid=grid(512), stream=stream0)
        buf1026 = reinterpret_tensor(buf1018, (8, 3136, 128), (401408, 128, 1), 0); del buf1018  # reuse
        # Source Nodes: [div__1], Original ATen: [aten.div, aten.mul]
        triton_poi_fused_div_mul_109.run(buf1025, bernoulli_1, buf1026, 3211264, grid=grid(3211264), stream=stream0)
        del bernoulli_1
        buf1027 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1026, (25088, 128), (128, 1), 0), permute_839, out=buf1027)
        del permute_839
        buf1028 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1026, (128, 25088), (1, 128), 0), view_51, out=buf1028)
        del view_51
        buf1029 = reinterpret_tensor(buf1023, (1, 128, 196), (25088, 1, 128), 0); del buf1023  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_110.run(buf1026, buf1029, 25088, 128, grid=grid(25088), stream=stream0)
        buf1030 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_111.run(buf1029, buf1030, 128, 196, grid=grid(128), stream=stream0)
        buf1031 = reinterpret_tensor(buf1027, (8, 3136, 512), (1605632, 512, 1), 0); del buf1027  # reuse
        # Source Nodes: [x_35], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_112.run(buf1031, addmm_6, 12845056, grid=grid(12845056), stream=stream0)
        del addmm_6
        buf1032 = reinterpret_tensor(buf1026, (25088, 128), (128, 1), 0); del buf1026  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1031, (25088, 512), (512, 1), 0), permute_843, out=buf1032)
        del permute_843
        buf1033 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1031, (512, 25088), (1, 512), 0), view_49, out=buf1033)
        del view_49
        buf1034 = empty_strided((1, 512, 196), (100352, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_113.run(buf1031, buf1034, 100352, 128, grid=grid(100352), stream=stream0)
        buf1035 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_114.run(buf1034, buf1035, 512, 196, grid=grid(512), stream=stream0)
        buf1042 = empty((8, 3136, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_115.run(buf1032, primals_47, mul_14, buf1025, div_119, buf1042, 25088, 128, grid=grid(25088), stream=stream0)
        del div_119
        del primals_47
        buf1038 = reinterpret_tensor(buf1029, (128, 196), (1, 128), 0); del buf1029  # reuse
        buf1040 = reinterpret_tensor(buf1021, (128, 196), (1, 128), 0); del buf1021  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_116.run(buf1032, mul_14, buf1038, buf1040, 25088, 128, grid=grid(25088), stream=stream0)
        del mul_14
        buf1039 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_111.run(buf1038, buf1039, 128, 196, grid=grid(128), stream=stream0)
        buf1041 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_111.run(buf1040, buf1041, 128, 196, grid=grid(128), stream=stream0)
        buf1043 = reinterpret_tensor(buf1032, (8, 8, 8, 7, 7, 128), (401408, 50176, 6272, 896, 128, 1), 0); del buf1032  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_117.run(buf1042, bernoulli, buf1043, 3211264, grid=grid(3211264), stream=stream0)
        del bernoulli
        buf1044 = reinterpret_tensor(buf1025, (25088, 128), (128, 1), 0); del buf1025  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1043, (25088, 128), (128, 1), 0), permute_848, out=buf1044)
        del permute_848
        buf1045 = reinterpret_tensor(buf62, (128, 128), (128, 1), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1043, (128, 25088), (1, 128), 0), view_43, out=buf1045)
        del view_43
        buf1046 = reinterpret_tensor(buf1040, (1, 128, 196), (25088, 1, 128), 0); del buf1040  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_110.run(buf1043, buf1046, 25088, 128, grid=grid(25088), stream=stream0)
        buf1047 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_111.run(buf1046, buf1047, 128, 196, grid=grid(128), stream=stream0)
        buf1048 = reinterpret_tensor(buf1043, (512, 4, 49, 32), (6272, 1568, 32, 1), 0); del buf1043  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_118.run(buf1044, buf1048, 3211264, grid=grid(3211264), stream=stream0)
        buf1049 = reinterpret_tensor(buf1044, (2048, 49, 32), (1568, 32, 1), 0); del buf1044  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_853, reinterpret_tensor(buf1048, (2048, 49, 32), (1568, 32, 1), 0), out=buf1049)
        del permute_853
        buf1050 = empty((2048, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1048, (2048, 49, 32), (1568, 32, 1), 0), permute_854, out=buf1050)
        del permute_854
        buf1051 = reinterpret_tensor(buf1034, (512, 4, 49, 1), (196, 49, 1, 100352), 0); del buf1034  # reuse
        buf1056 = empty((512, 4, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_119.run(buf1050, alias_46, buf1051, buf1056, 100352, 49, grid=grid(100352), stream=stream0)
        buf1052 = empty((1, 4, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_120.run(buf1050, alias_46, buf1051, buf1052, 9604, 512, grid=grid(9604), stream=stream0)
        del alias_46
        buf1053 = empty((169, 4), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_121.run(buf1053, 676, grid=grid(676), stream=stream0)
        aten.index_put_(buf1053, [view_35], reinterpret_tensor(buf1052, (2401, 4), (1, 2401), 0), True)
        del view_35
        buf1057 = reinterpret_tensor(buf1048, (2048, 32, 49), (1568, 49, 1), 0); del buf1048  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_856, reinterpret_tensor(buf1056, (2048, 49, 49), (2401, 49, 1), 0), out=buf1057)
        del permute_856
        buf1058 = empty((2048, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1056, (2048, 49, 49), (2401, 49, 1), 0), permute_857, out=buf1058)
        del permute_857
        buf1059 = empty((512, 49, 3, 4, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_122.run(buf1058, buf1057, buf1049, buf1059, 301056, 32, grid=grid(301056, 32), stream=stream0)
        buf1060 = reinterpret_tensor(buf1058, (25088, 128), (128, 1), 0); del buf1058  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1059, (25088, 384), (384, 1), 0), permute_860, out=buf1060)
        del permute_860
        buf1061 = empty((384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1059, (384, 25088), (1, 384), 0), view_29, out=buf1061)
        del view_29
        buf1062 = empty_strided((1, 384, 196), (75264, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_123.run(buf1059, buf1062, 75264, 128, grid=grid(75264), stream=stream0)
        buf1063 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_124.run(buf1062, buf1063, 384, 196, grid=grid(384), stream=stream0)
        buf1070 = reinterpret_tensor(buf1042, (8, 56, 56, 128), (401408, 7168, 128, 1), 0); del buf1042  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.roll]
        triton_red_fused_add_native_layer_norm_backward_roll_125.run(buf1070, buf1060, primals_41, mul_10, div_120, 25088, 128, grid=grid(25088), stream=stream0)
        del div_120
        del primals_41
        buf1066 = reinterpret_tensor(buf1046, (128, 196), (196, 1), 0); del buf1046  # reuse
        buf1068 = reinterpret_tensor(buf1038, (128, 196), (196, 1), 0); del buf1038  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_red_fused_native_layer_norm_backward_roll_126.run(buf1060, mul_10, buf1066, buf1068, 25088, 128, grid=grid(25088), stream=stream0)
        del mul_10
        buf1067 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_127.run(buf1066, buf1067, 128, 196, grid=grid(128), stream=stream0)
        buf1069 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.roll]
        triton_per_fused_native_layer_norm_backward_roll_127.run(buf1068, buf1069, 128, 196, grid=grid(128), stream=stream0)
        buf1071 = reinterpret_tensor(buf1031, (25088, 512), (512, 1), 0); del buf1031  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1070, (25088, 128), (128, 1), 0), permute_865, out=buf1071)
        del permute_865
        buf1072 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1070, (128, 25088), (1, 128), 0), view_23, out=buf1072)
        del view_23
        buf1073 = reinterpret_tensor(buf1068, (1, 128, 196), (25088, 1, 128), 0); del buf1068  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_128.run(buf1070, buf1073, 25088, 128, grid=grid(25088), stream=stream0)
        buf1074 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_111.run(buf1073, buf1074, 128, 196, grid=grid(128), stream=stream0)
        buf1075 = reinterpret_tensor(buf1071, (8, 3136, 512), (1605632, 512, 1), 0); del buf1071  # reuse
        # Source Nodes: [x_17], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_112.run(buf1075, addmm_2, 12845056, grid=grid(12845056), stream=stream0)
        del addmm_2
        buf1076 = buf1060; del buf1060  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1075, (25088, 512), (512, 1), 0), permute_869, out=buf1076)
        del permute_869
        buf1077 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1075, (512, 25088), (1, 512), 0), view_21, out=buf1077)
        del view_21
        buf1078 = reinterpret_tensor(buf1051, (1, 512, 196), (100352, 1, 512), 0); del buf1051  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_113.run(buf1075, buf1078, 100352, 128, grid=grid(100352), stream=stream0)
        del buf1075
        buf1079 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_114.run(buf1078, buf1079, 512, 196, grid=grid(512), stream=stream0)
        buf1086 = reinterpret_tensor(buf1070, (8, 3136, 128), (401408, 128, 1), 0); del buf1070  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_129.run(buf1086, buf1076, primals_35, mul_5, div_121, 25088, 128, grid=grid(25088), stream=stream0)
        del div_121
        del primals_35
        buf1082 = reinterpret_tensor(buf1073, (128, 196), (1, 128), 0); del buf1073  # reuse
        buf1084 = reinterpret_tensor(buf1066, (128, 196), (1, 128), 0); del buf1066  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_116.run(buf1076, mul_5, buf1082, buf1084, 25088, 128, grid=grid(25088), stream=stream0)
        del mul_5
        buf1083 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_111.run(buf1082, buf1083, 128, 196, grid=grid(128), stream=stream0)
        buf1085 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_111.run(buf1084, buf1085, 128, 196, grid=grid(128), stream=stream0)
        buf1087 = buf1076; del buf1076  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_130.run(buf1086, buf1087, 3211264, grid=grid(3211264), stream=stream0)
        buf1088 = reinterpret_tensor(buf1057, (25088, 128), (128, 1), 0); del buf1057  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1087, permute_874, out=buf1088)
        del permute_874
        buf1089 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1087, (128, 25088), (1, 128), 0), view_15, out=buf1089)
        del view_15
        buf1090 = reinterpret_tensor(buf1084, (1, 128, 196), (25088, 1, 128), 0); del buf1084  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_110.run(buf1087, buf1090, 25088, 128, grid=grid(25088), stream=stream0)
        buf1091 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_111.run(buf1090, buf1091, 128, 196, grid=grid(128), stream=stream0)
        buf1092 = reinterpret_tensor(buf1087, (512, 4, 49, 32), (6272, 1568, 32, 1), 0); del buf1087  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_118.run(buf1088, buf1092, 3211264, grid=grid(3211264), stream=stream0)
        buf1093 = reinterpret_tensor(buf1088, (2048, 49, 32), (1568, 32, 1), 0); del buf1088  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_879, reinterpret_tensor(buf1092, (2048, 49, 32), (1568, 32, 1), 0), out=buf1093)
        del permute_879
        buf1094 = reinterpret_tensor(buf1056, (2048, 49, 49), (2401, 49, 1), 0); del buf1056  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1092, (2048, 49, 32), (1568, 32, 1), 0), permute_880, out=buf1094)
        del permute_880
        buf1095 = reinterpret_tensor(buf1078, (512, 4, 49, 1), (196, 49, 1, 100352), 0); del buf1078  # reuse
        buf1100 = reinterpret_tensor(buf1050, (512, 4, 49, 49), (9604, 2401, 49, 1), 0); del buf1050  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_119.run(buf1094, alias_47, buf1095, buf1100, 100352, 49, grid=grid(100352), stream=stream0)
        buf1096 = buf1052; del buf1052  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_red_fused_sum_120.run(buf1094, alias_47, buf1095, buf1096, 9604, 512, grid=grid(9604), stream=stream0)
        del alias_47
        del buf1094
        del buf1095
        buf1097 = empty((169, 4), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_121.run(buf1097, 676, grid=grid(676), stream=stream0)
        aten.index_put_(buf1097, [view_9], reinterpret_tensor(buf1096, (2401, 4), (1, 2401), 0), True)
        del buf1096
        del view_9
        buf1101 = reinterpret_tensor(buf1092, (2048, 32, 49), (1568, 49, 1), 0); del buf1092  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_882, reinterpret_tensor(buf1100, (2048, 49, 49), (2401, 49, 1), 0), out=buf1101)
        del permute_882
        buf1102 = buf1049; del buf1049  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1100, (2048, 49, 49), (2401, 49, 1), 0), permute_883, out=buf1102)
        del buf1100
        del permute_883
        buf1103 = buf1059; del buf1059  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_122.run(buf1102, buf1101, buf1093, buf1103, 301056, 32, grid=grid(301056, 32), stream=stream0)
        del buf1093
        buf1104 = reinterpret_tensor(buf1102, (25088, 128), (128, 1), 0); del buf1102  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1103, (25088, 384), (384, 1), 0), permute_886, out=buf1104)
        del permute_886
        buf1105 = empty((384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1103, (384, 25088), (1, 384), 0), view_3, out=buf1105)
        del view_3
        buf1106 = buf1062; del buf1062  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_123.run(buf1103, buf1106, 75264, 128, grid=grid(75264), stream=stream0)
        del buf1103
        buf1107 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_124.run(buf1106, buf1107, 384, 196, grid=grid(384), stream=stream0)
        del buf1106
        buf1114 = reinterpret_tensor(buf1086, (8, 56, 56, 128), (401408, 7168, 128, 1), 0); del buf1086  # reuse
        buf1121 = reinterpret_tensor(buf1101, (8, 56, 56, 128), (401408, 7168, 128, 1), 0); del buf1101  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_131.run(buf1114, buf1104, primals_29, mul_2, div_122, primals_27, mul, div_123, buf1121, 25088, 128, grid=grid(25088), stream=stream0)
        del div_122
        del div_123
        del primals_27
        del primals_29
        buf1110 = reinterpret_tensor(buf1090, (128, 196), (1, 128), 0); del buf1090  # reuse
        buf1112 = buf1082; del buf1082  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_132.run(buf1104, mul_2, buf1110, buf1112, 25088, 128, grid=grid(25088), stream=stream0)
        del buf1104
        del mul_2
        buf1111 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_111.run(buf1110, buf1111, 128, 196, grid=grid(128), stream=stream0)
        buf1113 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_111.run(buf1112, buf1113, 128, 196, grid=grid(128), stream=stream0)
        buf1117 = buf1112; del buf1112  # reuse
        buf1119 = buf1110; del buf1110  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_116.run(buf1114, mul, buf1117, buf1119, 25088, 128, grid=grid(25088), stream=stream0)
        del buf1114
        del mul
        buf1118 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_111.run(buf1117, buf1118, 128, 196, grid=grid(128), stream=stream0)
        del buf1117
        buf1120 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_111.run(buf1119, buf1120, 128, 196, grid=grid(128), stream=stream0)
        buf1122 = buf1119; del buf1119  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_sum_128.run(buf1121, buf1122, 25088, 128, grid=grid(25088), stream=stream0)
        buf1123 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_sum_111.run(buf1122, buf1123, 128, 196, grid=grid(128), stream=stream0)
        del buf1122
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1124 = aten.convolution_backward(reinterpret_tensor(buf1121, (8, 128, 56, 56), (401408, 1, 7168, 128), 0), primals_365, primals_25, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf1121
        del primals_25
        del primals_365
        buf1125 = buf1124[1]
        return (buf1097, buf1053, buf999, buf954, buf900, buf855, buf810, buf765, buf720, buf675, buf630, buf585, buf540, buf495, buf450, buf405, buf360, buf315, buf270, buf225, buf180, buf135, buf81, buf36, buf1125, buf1123, buf1118, buf1120, buf1111, buf1113, reinterpret_tensor(buf1105, (384, 128), (128, 1), 0), reinterpret_tensor(buf1107, (384, ), (1, ), 0), reinterpret_tensor(buf1089, (128, 128), (128, 1), 0), reinterpret_tensor(buf1091, (128, ), (1, ), 0), buf1083, buf1085, reinterpret_tensor(buf1077, (512, 128), (128, 1), 0), reinterpret_tensor(buf1079, (512, ), (1, ), 0), reinterpret_tensor(buf1072, (128, 512), (512, 1), 0), reinterpret_tensor(buf1074, (128, ), (1, ), 0), buf1067, buf1069, reinterpret_tensor(buf1061, (384, 128), (128, 1), 0), reinterpret_tensor(buf1063, (384, ), (1, ), 0), reinterpret_tensor(buf1045, (128, 128), (128, 1), 0), reinterpret_tensor(buf1047, (128, ), (1, ), 0), buf1039, buf1041, reinterpret_tensor(buf1033, (512, 128), (128, 1), 0), reinterpret_tensor(buf1035, (512, ), (1, ), 0), reinterpret_tensor(buf1028, (128, 512), (512, 1), 0), reinterpret_tensor(buf1030, (128, ), (1, ), 0), buf1022, buf1024, reinterpret_tensor(buf1017, (256, 512), (512, 1), 0), buf1013, buf1015, reinterpret_tensor(buf1007, (768, 256), (256, 1), 0), reinterpret_tensor(buf1009, (768, ), (1, ), 0), reinterpret_tensor(buf991, (256, 256), (256, 1), 0), reinterpret_tensor(buf993, (256, ), (1, ), 0), buf985, buf987, reinterpret_tensor(buf979, (1024, 256), (256, 1), 0), reinterpret_tensor(buf981, (1024, ), (1, ), 0), reinterpret_tensor(buf974, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf976, (256, ), (1, ), 0), buf968, buf970, reinterpret_tensor(buf962, (768, 256), (256, 1), 0), reinterpret_tensor(buf964, (768, ), (1, ), 0), reinterpret_tensor(buf946, (256, 256), (256, 1), 0), reinterpret_tensor(buf948, (256, ), (1, ), 0), buf940, buf942, reinterpret_tensor(buf934, (1024, 256), (256, 1), 0), reinterpret_tensor(buf936, (1024, ), (1, ), 0), reinterpret_tensor(buf929, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf931, (256, ), (1, ), 0), buf923, buf925, reinterpret_tensor(buf918, (512, 1024), (1024, 1), 0), buf914, buf916, reinterpret_tensor(buf908, (1536, 512), (512, 1), 0), reinterpret_tensor(buf910, (1536, ), (1, ), 0), reinterpret_tensor(buf892, (512, 512), (512, 1), 0), reinterpret_tensor(buf894, (512, ), (1, ), 0), buf886, buf888, reinterpret_tensor(buf880, (2048, 512), (512, 1), 0), reinterpret_tensor(buf882, (2048, ), (1, ), 0), reinterpret_tensor(buf875, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf877, (512, ), (1, ), 0), buf869, buf871, reinterpret_tensor(buf863, (1536, 512), (512, 1), 0), reinterpret_tensor(buf865, (1536, ), (1, ), 0), reinterpret_tensor(buf847, (512, 512), (512, 1), 0), reinterpret_tensor(buf849, (512, ), (1, ), 0), buf841, buf843, reinterpret_tensor(buf835, (2048, 512), (512, 1), 0), reinterpret_tensor(buf837, (2048, ), (1, ), 0), reinterpret_tensor(buf830, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf832, (512, ), (1, ), 0), buf824, buf826, reinterpret_tensor(buf818, (1536, 512), (512, 1), 0), reinterpret_tensor(buf820, (1536, ), (1, ), 0), reinterpret_tensor(buf802, (512, 512), (512, 1), 0), reinterpret_tensor(buf804, (512, ), (1, ), 0), buf796, buf798, reinterpret_tensor(buf790, (2048, 512), (512, 1), 0), reinterpret_tensor(buf792, (2048, ), (1, ), 0), reinterpret_tensor(buf785, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf787, (512, ), (1, ), 0), buf779, buf781, reinterpret_tensor(buf773, (1536, 512), (512, 1), 0), reinterpret_tensor(buf775, (1536, ), (1, ), 0), reinterpret_tensor(buf757, (512, 512), (512, 1), 0), reinterpret_tensor(buf759, (512, ), (1, ), 0), buf751, buf753, reinterpret_tensor(buf745, (2048, 512), (512, 1), 0), reinterpret_tensor(buf747, (2048, ), (1, ), 0), reinterpret_tensor(buf740, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf742, (512, ), (1, ), 0), buf734, buf736, reinterpret_tensor(buf728, (1536, 512), (512, 1), 0), reinterpret_tensor(buf730, (1536, ), (1, ), 0), reinterpret_tensor(buf712, (512, 512), (512, 1), 0), reinterpret_tensor(buf714, (512, ), (1, ), 0), buf706, buf708, reinterpret_tensor(buf700, (2048, 512), (512, 1), 0), reinterpret_tensor(buf702, (2048, ), (1, ), 0), reinterpret_tensor(buf695, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf697, (512, ), (1, ), 0), buf689, buf691, reinterpret_tensor(buf683, (1536, 512), (512, 1), 0), reinterpret_tensor(buf685, (1536, ), (1, ), 0), reinterpret_tensor(buf667, (512, 512), (512, 1), 0), reinterpret_tensor(buf669, (512, ), (1, ), 0), buf661, buf663, reinterpret_tensor(buf655, (2048, 512), (512, 1), 0), reinterpret_tensor(buf657, (2048, ), (1, ), 0), reinterpret_tensor(buf650, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf652, (512, ), (1, ), 0), buf644, buf646, reinterpret_tensor(buf638, (1536, 512), (512, 1), 0), reinterpret_tensor(buf640, (1536, ), (1, ), 0), reinterpret_tensor(buf622, (512, 512), (512, 1), 0), reinterpret_tensor(buf624, (512, ), (1, ), 0), buf616, buf618, reinterpret_tensor(buf610, (2048, 512), (512, 1), 0), reinterpret_tensor(buf612, (2048, ), (1, ), 0), reinterpret_tensor(buf605, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf607, (512, ), (1, ), 0), buf599, buf601, reinterpret_tensor(buf593, (1536, 512), (512, 1), 0), reinterpret_tensor(buf595, (1536, ), (1, ), 0), reinterpret_tensor(buf577, (512, 512), (512, 1), 0), reinterpret_tensor(buf579, (512, ), (1, ), 0), buf571, buf573, reinterpret_tensor(buf565, (2048, 512), (512, 1), 0), reinterpret_tensor(buf567, (2048, ), (1, ), 0), reinterpret_tensor(buf560, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf562, (512, ), (1, ), 0), buf554, buf556, reinterpret_tensor(buf548, (1536, 512), (512, 1), 0), reinterpret_tensor(buf550, (1536, ), (1, ), 0), reinterpret_tensor(buf532, (512, 512), (512, 1), 0), reinterpret_tensor(buf534, (512, ), (1, ), 0), buf526, buf528, reinterpret_tensor(buf520, (2048, 512), (512, 1), 0), reinterpret_tensor(buf522, (2048, ), (1, ), 0), reinterpret_tensor(buf515, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf517, (512, ), (1, ), 0), buf509, buf511, reinterpret_tensor(buf503, (1536, 512), (512, 1), 0), reinterpret_tensor(buf505, (1536, ), (1, ), 0), reinterpret_tensor(buf487, (512, 512), (512, 1), 0), reinterpret_tensor(buf489, (512, ), (1, ), 0), buf481, buf483, reinterpret_tensor(buf475, (2048, 512), (512, 1), 0), reinterpret_tensor(buf477, (2048, ), (1, ), 0), reinterpret_tensor(buf470, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf472, (512, ), (1, ), 0), buf464, buf466, reinterpret_tensor(buf458, (1536, 512), (512, 1), 0), reinterpret_tensor(buf460, (1536, ), (1, ), 0), reinterpret_tensor(buf442, (512, 512), (512, 1), 0), reinterpret_tensor(buf444, (512, ), (1, ), 0), buf436, buf438, reinterpret_tensor(buf430, (2048, 512), (512, 1), 0), reinterpret_tensor(buf432, (2048, ), (1, ), 0), reinterpret_tensor(buf425, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf427, (512, ), (1, ), 0), buf419, buf421, reinterpret_tensor(buf413, (1536, 512), (512, 1), 0), reinterpret_tensor(buf415, (1536, ), (1, ), 0), reinterpret_tensor(buf397, (512, 512), (512, 1), 0), reinterpret_tensor(buf399, (512, ), (1, ), 0), buf391, buf393, reinterpret_tensor(buf385, (2048, 512), (512, 1), 0), reinterpret_tensor(buf387, (2048, ), (1, ), 0), reinterpret_tensor(buf380, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf382, (512, ), (1, ), 0), buf374, buf376, reinterpret_tensor(buf368, (1536, 512), (512, 1), 0), reinterpret_tensor(buf370, (1536, ), (1, ), 0), reinterpret_tensor(buf352, (512, 512), (512, 1), 0), reinterpret_tensor(buf354, (512, ), (1, ), 0), buf346, buf348, reinterpret_tensor(buf340, (2048, 512), (512, 1), 0), reinterpret_tensor(buf342, (2048, ), (1, ), 0), reinterpret_tensor(buf335, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf337, (512, ), (1, ), 0), buf329, buf331, reinterpret_tensor(buf323, (1536, 512), (512, 1), 0), reinterpret_tensor(buf325, (1536, ), (1, ), 0), reinterpret_tensor(buf307, (512, 512), (512, 1), 0), reinterpret_tensor(buf309, (512, ), (1, ), 0), buf301, buf303, reinterpret_tensor(buf295, (2048, 512), (512, 1), 0), reinterpret_tensor(buf297, (2048, ), (1, ), 0), reinterpret_tensor(buf290, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf292, (512, ), (1, ), 0), buf284, buf286, reinterpret_tensor(buf278, (1536, 512), (512, 1), 0), reinterpret_tensor(buf280, (1536, ), (1, ), 0), reinterpret_tensor(buf262, (512, 512), (512, 1), 0), reinterpret_tensor(buf264, (512, ), (1, ), 0), buf256, buf258, reinterpret_tensor(buf250, (2048, 512), (512, 1), 0), reinterpret_tensor(buf252, (2048, ), (1, ), 0), reinterpret_tensor(buf245, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf247, (512, ), (1, ), 0), buf239, buf241, reinterpret_tensor(buf233, (1536, 512), (512, 1), 0), reinterpret_tensor(buf235, (1536, ), (1, ), 0), reinterpret_tensor(buf217, (512, 512), (512, 1), 0), reinterpret_tensor(buf219, (512, ), (1, ), 0), buf211, buf213, reinterpret_tensor(buf205, (2048, 512), (512, 1), 0), reinterpret_tensor(buf207, (2048, ), (1, ), 0), reinterpret_tensor(buf200, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf202, (512, ), (1, ), 0), buf194, buf196, reinterpret_tensor(buf188, (1536, 512), (512, 1), 0), reinterpret_tensor(buf190, (1536, ), (1, ), 0), reinterpret_tensor(buf172, (512, 512), (512, 1), 0), reinterpret_tensor(buf174, (512, ), (1, ), 0), buf166, buf168, reinterpret_tensor(buf160, (2048, 512), (512, 1), 0), reinterpret_tensor(buf162, (2048, ), (1, ), 0), reinterpret_tensor(buf155, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf157, (512, ), (1, ), 0), buf149, buf151, reinterpret_tensor(buf143, (1536, 512), (512, 1), 0), reinterpret_tensor(buf145, (1536, ), (1, ), 0), reinterpret_tensor(buf127, (512, 512), (512, 1), 0), reinterpret_tensor(buf129, (512, ), (1, ), 0), buf121, buf123, reinterpret_tensor(buf115, (2048, 512), (512, 1), 0), reinterpret_tensor(buf117, (2048, ), (1, ), 0), reinterpret_tensor(buf110, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf112, (512, ), (1, ), 0), buf104, buf106, reinterpret_tensor(buf99, (1024, 2048), (2048, 1), 0), buf95, buf97, reinterpret_tensor(buf89, (3072, 1024), (1024, 1), 0), reinterpret_tensor(buf91, (3072, ), (1, ), 0), reinterpret_tensor(buf73, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf75, (1024, ), (1, ), 0), buf67, buf69, reinterpret_tensor(buf61, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf63, (4096, ), (1, ), 0), reinterpret_tensor(buf56, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf58, (1024, ), (1, ), 0), buf50, buf52, reinterpret_tensor(buf44, (3072, 1024), (1024, 1), 0), reinterpret_tensor(buf46, (3072, ), (1, ), 0), reinterpret_tensor(buf28, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf30, (1024, ), (1, ), 0), buf22, buf24, reinterpret_tensor(buf16, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf18, (4096, ), (1, ), 0), reinterpret_tensor(buf11, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf13, (1024, ), (1, ), 0), buf6, buf7, reinterpret_tensor(buf1, (1000, 1024), (1024, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_25 = rand_strided((128, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    mul = rand_strided((8, 56, 56, 128), (401408, 7168, 128, 1), device='cuda:0', dtype=torch.float32)
    mul_2 = rand_strided((8, 56, 56, 128), (401408, 7168, 128, 1), device='cuda:0', dtype=torch.float32)
    view_3 = rand_strided((25088, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_9 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_15 = rand_strided((25088, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_5 = rand_strided((8, 3136, 128), (401408, 128, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((25088, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mul_10 = rand_strided((8, 56, 56, 128), (401408, 7168, 128, 1), device='cuda:0', dtype=torch.float32)
    view_29 = rand_strided((25088, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_35 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_43 = rand_strided((25088, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    bernoulli = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_14 = rand_strided((8, 3136, 128), (401408, 128, 1), device='cuda:0', dtype=torch.float32)
    view_49 = rand_strided((25088, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_6 = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_51 = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_1 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_20 = rand_strided((8, 28, 28, 512), (401408, 14336, 512, 1), device='cuda:0', dtype=torch.float32)
    view_56 = rand_strided((6272, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm = rand_strided((6272, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_19 = rand_strided((8, 28, 28, 1), (784, 28, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_6 = rand_strided((8, 28, 28, 1), (784, 28, 1, 1), device='cuda:0', dtype=torch.float32)
    view_61 = rand_strided((6272, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    view_67 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_73 = rand_strided((6272, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_2 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_26 = rand_strided((8, 784, 256), (200704, 256, 1), device='cuda:0', dtype=torch.float32)
    view_79 = rand_strided((6272, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_81 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_3 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_32 = rand_strided((8, 28, 28, 256), (200704, 7168, 256, 1), device='cuda:0', dtype=torch.float32)
    view_87 = rand_strided((6272, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    view_93 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_101 = rand_strided((6272, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_4 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_36 = rand_strided((8, 784, 256), (200704, 256, 1), device='cuda:0', dtype=torch.float32)
    view_107 = rand_strided((6272, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_109 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_5 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_42 = rand_strided((8, 14, 14, 1024), (200704, 14336, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_114 = rand_strided((1568, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mm_1 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_35 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_11 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    view_119 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_125 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_131 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_6 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_48 = rand_strided((8, 196, 512), (100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_137 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_18 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_139 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_7 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_54 = rand_strided((8, 14, 14, 512), (100352, 7168, 512, 1), device='cuda:0', dtype=torch.float32)
    view_145 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_151 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_159 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_8 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_58 = rand_strided((8, 196, 512), (100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_165 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_167 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_9 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_64 = rand_strided((8, 14, 14, 512), (100352, 7168, 512, 1), device='cuda:0', dtype=torch.float32)
    view_173 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_179 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_185 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_10 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_68 = rand_strided((8, 196, 512), (100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_191 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_26 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_193 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_11 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_74 = rand_strided((8, 14, 14, 512), (100352, 7168, 512, 1), device='cuda:0', dtype=torch.float32)
    view_199 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_205 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_213 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_12 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_78 = rand_strided((8, 196, 512), (100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_219 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_30 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_221 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_13 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_84 = rand_strided((8, 14, 14, 512), (100352, 7168, 512, 1), device='cuda:0', dtype=torch.float32)
    view_227 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_233 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_239 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_14 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_88 = rand_strided((8, 196, 512), (100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_245 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_247 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_15 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_94 = rand_strided((8, 14, 14, 512), (100352, 7168, 512, 1), device='cuda:0', dtype=torch.float32)
    view_253 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_259 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_267 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_16 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_98 = rand_strided((8, 196, 512), (100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_273 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_38 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_275 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_17 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_104 = rand_strided((8, 14, 14, 512), (100352, 7168, 512, 1), device='cuda:0', dtype=torch.float32)
    view_281 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_287 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_293 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_18 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_108 = rand_strided((8, 196, 512), (100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_299 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_42 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_301 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_19 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_114 = rand_strided((8, 14, 14, 512), (100352, 7168, 512, 1), device='cuda:0', dtype=torch.float32)
    view_307 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_313 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_321 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_20 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_118 = rand_strided((8, 196, 512), (100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_327 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_329 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_21 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_124 = rand_strided((8, 14, 14, 512), (100352, 7168, 512, 1), device='cuda:0', dtype=torch.float32)
    view_335 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_341 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_347 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_22 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_128 = rand_strided((8, 196, 512), (100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_353 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_50 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_355 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_23 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_134 = rand_strided((8, 14, 14, 512), (100352, 7168, 512, 1), device='cuda:0', dtype=torch.float32)
    view_361 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_367 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_375 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_24 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_138 = rand_strided((8, 196, 512), (100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_381 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_54 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_383 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_25 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_144 = rand_strided((8, 14, 14, 512), (100352, 7168, 512, 1), device='cuda:0', dtype=torch.float32)
    view_389 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_395 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_401 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_26 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_148 = rand_strided((8, 196, 512), (100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_407 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_58 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_409 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_27 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_154 = rand_strided((8, 14, 14, 512), (100352, 7168, 512, 1), device='cuda:0', dtype=torch.float32)
    view_415 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_421 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_429 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_28 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_158 = rand_strided((8, 196, 512), (100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_435 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_62 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_437 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_29 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_164 = rand_strided((8, 14, 14, 512), (100352, 7168, 512, 1), device='cuda:0', dtype=torch.float32)
    view_443 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_449 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_455 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_30 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_168 = rand_strided((8, 196, 512), (100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_461 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_66 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_463 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_31 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_174 = rand_strided((8, 14, 14, 512), (100352, 7168, 512, 1), device='cuda:0', dtype=torch.float32)
    view_469 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_475 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_483 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_32 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_178 = rand_strided((8, 196, 512), (100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_489 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_70 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_491 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_33 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_184 = rand_strided((8, 14, 14, 512), (100352, 7168, 512, 1), device='cuda:0', dtype=torch.float32)
    view_497 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_503 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_509 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_34 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_188 = rand_strided((8, 196, 512), (100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_515 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_74 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_517 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_35 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_194 = rand_strided((8, 14, 14, 512), (100352, 7168, 512, 1), device='cuda:0', dtype=torch.float32)
    view_523 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_529 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_537 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_36 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_198 = rand_strided((8, 196, 512), (100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_543 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_78 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_545 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_37 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_204 = rand_strided((8, 14, 14, 512), (100352, 7168, 512, 1), device='cuda:0', dtype=torch.float32)
    view_551 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_557 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_563 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_38 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_208 = rand_strided((8, 196, 512), (100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_569 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_82 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_571 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_39 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_214 = rand_strided((8, 14, 14, 512), (100352, 7168, 512, 1), device='cuda:0', dtype=torch.float32)
    view_577 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_583 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_591 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_40 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_218 = rand_strided((8, 196, 512), (100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_597 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_86 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_599 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_41 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_224 = rand_strided((8, 7, 7, 2048), (100352, 14336, 2048, 1), device='cuda:0', dtype=torch.float32)
    view_604 = rand_strided((392, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mm_2 = rand_strided((392, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_163 = rand_strided((8, 7, 7, 1), (49, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_48 = rand_strided((8, 7, 7, 1), (49, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    view_609 = rand_strided((392, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_615 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_621 = rand_strided((392, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_42 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_230 = rand_strided((8, 49, 1024), (50176, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_627 = rand_strided((392, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_90 = rand_strided((392, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_629 = rand_strided((392, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_43 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_236 = rand_strided((8, 7, 7, 1024), (50176, 7168, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_635 = rand_strided((392, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_641 = rand_strided((2401, ), (1, ), device='cuda:0', dtype=torch.int64)
    view_647 = rand_strided((392, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_44 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_240 = rand_strided((8, 49, 1024), (50176, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_653 = rand_strided((392, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_94 = rand_strided((392, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_655 = rand_strided((392, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_45 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_246 = rand_strided((8, 7, 7, 1024), (50176, 7168, 1024, 1), device='cuda:0', dtype=torch.float32)
    clone_264 = rand_strided((8, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_248 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_71 = rand_strided((8, 7, 7, 1), (49, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_252 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_256 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_72 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_261 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_266 = rand_strided((256, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_267 = rand_strided((256, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_24 = rand_strided((8, 32, 49, 49), (76832, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_269 = rand_strided((256, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_270 = rand_strided((256, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_273 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_73 = rand_strided((8, 7, 7, 1), (49, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_278 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_282 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_74 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_287 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_292 = rand_strided((256, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_293 = rand_strided((256, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_25 = rand_strided((8, 32, 49, 49), (76832, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_295 = rand_strided((256, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_296 = rand_strided((256, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_299 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_306 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    div_76 = rand_strided((8, 7, 7, 1), (49, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_309 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_313 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_77 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_318 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_323 = rand_strided((512, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_324 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_26 = rand_strided((32, 16, 49, 49), (38416, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_326 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_327 = rand_strided((512, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_330 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_78 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_335 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_339 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_79 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_344 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_349 = rand_strided((512, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_350 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_27 = rand_strided((32, 16, 49, 49), (38416, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_352 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_353 = rand_strided((512, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_356 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_80 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_361 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_365 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_81 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_370 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_375 = rand_strided((512, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_376 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_28 = rand_strided((32, 16, 49, 49), (38416, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_378 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_379 = rand_strided((512, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_382 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_82 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_387 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_391 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_83 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_396 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_401 = rand_strided((512, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_402 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_29 = rand_strided((32, 16, 49, 49), (38416, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_404 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_405 = rand_strided((512, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_408 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_84 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_413 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_417 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_85 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_422 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_427 = rand_strided((512, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_428 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_30 = rand_strided((32, 16, 49, 49), (38416, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_430 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_431 = rand_strided((512, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_434 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_86 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_439 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_443 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_87 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_448 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_453 = rand_strided((512, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_454 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_31 = rand_strided((32, 16, 49, 49), (38416, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_456 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_457 = rand_strided((512, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_460 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_88 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_465 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_469 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_89 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_474 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_479 = rand_strided((512, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_480 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_32 = rand_strided((32, 16, 49, 49), (38416, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_482 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_483 = rand_strided((512, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_486 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_90 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_491 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_495 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_91 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_500 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_505 = rand_strided((512, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_506 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_33 = rand_strided((32, 16, 49, 49), (38416, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_508 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_509 = rand_strided((512, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_512 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_92 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_517 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_521 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_93 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_526 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_531 = rand_strided((512, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_532 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_34 = rand_strided((32, 16, 49, 49), (38416, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_534 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_535 = rand_strided((512, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_538 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_94 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_543 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_547 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_95 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_552 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_557 = rand_strided((512, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_558 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_35 = rand_strided((32, 16, 49, 49), (38416, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_560 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_561 = rand_strided((512, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_564 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_96 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_569 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_573 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_97 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_578 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_583 = rand_strided((512, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_584 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_36 = rand_strided((32, 16, 49, 49), (38416, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_586 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_587 = rand_strided((512, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_590 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_98 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_595 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_599 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_99 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_604 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_609 = rand_strided((512, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_610 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_37 = rand_strided((32, 16, 49, 49), (38416, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_612 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_613 = rand_strided((512, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_616 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_100 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_621 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_625 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_101 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_630 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_635 = rand_strided((512, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_636 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_38 = rand_strided((32, 16, 49, 49), (38416, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_638 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_639 = rand_strided((512, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_642 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_102 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_647 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_651 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_103 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_656 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_661 = rand_strided((512, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_662 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_39 = rand_strided((32, 16, 49, 49), (38416, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_664 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_665 = rand_strided((512, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_668 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_104 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_673 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_677 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_105 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_682 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_687 = rand_strided((512, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_688 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_40 = rand_strided((32, 16, 49, 49), (38416, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_690 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_691 = rand_strided((512, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_694 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_106 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_699 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_703 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_107 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_708 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_713 = rand_strided((512, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_714 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_41 = rand_strided((32, 16, 49, 49), (38416, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_716 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_717 = rand_strided((512, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_720 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_108 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_725 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_729 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_109 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_734 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_739 = rand_strided((512, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_740 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_42 = rand_strided((32, 16, 49, 49), (38416, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_742 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_743 = rand_strided((512, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_746 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_110 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_751 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_755 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_111 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_760 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_765 = rand_strided((512, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_766 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_43 = rand_strided((32, 16, 49, 49), (38416, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_768 = rand_strided((512, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_769 = rand_strided((512, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_772 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_779 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_113 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_782 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_786 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_114 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_791 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_796 = rand_strided((1024, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_797 = rand_strided((1024, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_44 = rand_strided((128, 8, 49, 49), (19208, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_799 = rand_strided((1024, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_800 = rand_strided((1024, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_803 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_115 = rand_strided((8, 28, 28, 1), (784, 28, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_808 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_812 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_116 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_817 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_822 = rand_strided((1024, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_823 = rand_strided((1024, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_45 = rand_strided((128, 8, 49, 49), (19208, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_825 = rand_strided((1024, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_826 = rand_strided((1024, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_829 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_836 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_118 = rand_strided((8, 28, 28, 1), (784, 28, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_839 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_843 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_119 = rand_strided((8, 3136, 1), (3136, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_848 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_853 = rand_strided((2048, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_854 = rand_strided((2048, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_46 = rand_strided((512, 4, 49, 49), (9604, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_856 = rand_strided((2048, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_857 = rand_strided((2048, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_860 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_120 = rand_strided((8, 56, 56, 1), (3136, 56, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_865 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_869 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_121 = rand_strided((8, 3136, 1), (3136, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_874 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_879 = rand_strided((2048, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_880 = rand_strided((2048, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_47 = rand_strided((512, 4, 49, 49), (9604, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_882 = rand_strided((2048, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_883 = rand_strided((2048, 49, 32), (1568, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_886 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_122 = rand_strided((8, 56, 56, 1), (3136, 56, 1, 1), device='cuda:0', dtype=torch.float32)
    div_123 = rand_strided((8, 56, 56, 1), (3136, 56, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_25, primals_27, primals_29, primals_35, primals_41, primals_47, primals_53, primals_56, primals_62, primals_68, primals_74, primals_80, primals_83, primals_89, primals_95, primals_101, primals_107, primals_113, primals_119, primals_125, primals_131, primals_137, primals_143, primals_149, primals_155, primals_161, primals_167, primals_173, primals_179, primals_185, primals_191, primals_197, primals_203, primals_209, primals_215, primals_221, primals_227, primals_233, primals_239, primals_245, primals_251, primals_257, primals_263, primals_269, primals_275, primals_281, primals_287, primals_293, primals_299, primals_302, primals_308, primals_314, primals_320, primals_326, primals_365, mul, mul_2, view_3, view_9, view_15, mul_5, view_21, addmm_2, view_23, mul_10, view_29, view_35, view_43, bernoulli, mul_14, view_49, addmm_6, view_51, bernoulli_1, mul_20, view_56, mm, getitem_19, rsqrt_6, view_61, view_67, view_73, bernoulli_2, mul_26, view_79, addmm_10, view_81, bernoulli_3, mul_32, view_87, view_93, view_101, bernoulli_4, mul_36, view_107, addmm_14, view_109, bernoulli_5, mul_42, view_114, mm_1, getitem_35, rsqrt_11, view_119, view_125, view_131, bernoulli_6, mul_48, view_137, addmm_18, view_139, bernoulli_7, mul_54, view_145, view_151, view_159, bernoulli_8, mul_58, view_165, addmm_22, view_167, bernoulli_9, mul_64, view_173, view_179, view_185, bernoulli_10, mul_68, view_191, addmm_26, view_193, bernoulli_11, mul_74, view_199, view_205, view_213, bernoulli_12, mul_78, view_219, addmm_30, view_221, bernoulli_13, mul_84, view_227, view_233, view_239, bernoulli_14, mul_88, view_245, addmm_34, view_247, bernoulli_15, mul_94, view_253, view_259, view_267, bernoulli_16, mul_98, view_273, addmm_38, view_275, bernoulli_17, mul_104, view_281, view_287, view_293, bernoulli_18, mul_108, view_299, addmm_42, view_301, bernoulli_19, mul_114, view_307, view_313, view_321, bernoulli_20, mul_118, view_327, addmm_46, view_329, bernoulli_21, mul_124, view_335, view_341, view_347, bernoulli_22, mul_128, view_353, addmm_50, view_355, bernoulli_23, mul_134, view_361, view_367, view_375, bernoulli_24, mul_138, view_381, addmm_54, view_383, bernoulli_25, mul_144, view_389, view_395, view_401, bernoulli_26, mul_148, view_407, addmm_58, view_409, bernoulli_27, mul_154, view_415, view_421, view_429, bernoulli_28, mul_158, view_435, addmm_62, view_437, bernoulli_29, mul_164, view_443, view_449, view_455, bernoulli_30, mul_168, view_461, addmm_66, view_463, bernoulli_31, mul_174, view_469, view_475, view_483, bernoulli_32, mul_178, view_489, addmm_70, view_491, bernoulli_33, mul_184, view_497, view_503, view_509, bernoulli_34, mul_188, view_515, addmm_74, view_517, bernoulli_35, mul_194, view_523, view_529, view_537, bernoulli_36, mul_198, view_543, addmm_78, view_545, bernoulli_37, mul_204, view_551, view_557, view_563, bernoulli_38, mul_208, view_569, addmm_82, view_571, bernoulli_39, mul_214, view_577, view_583, view_591, bernoulli_40, mul_218, view_597, addmm_86, view_599, bernoulli_41, mul_224, view_604, mm_2, getitem_163, rsqrt_48, view_609, view_615, view_621, bernoulli_42, mul_230, view_627, addmm_90, view_629, bernoulli_43, mul_236, view_635, view_641, view_647, bernoulli_44, mul_240, view_653, addmm_94, view_655, bernoulli_45, mul_246, clone_264, permute_248, div_71, permute_252, permute_256, div_72, permute_261, permute_266, permute_267, alias_24, permute_269, permute_270, permute_273, div_73, permute_278, permute_282, div_74, permute_287, permute_292, permute_293, alias_25, permute_295, permute_296, permute_299, permute_306, div_76, permute_309, permute_313, div_77, permute_318, permute_323, permute_324, alias_26, permute_326, permute_327, permute_330, div_78, permute_335, permute_339, div_79, permute_344, permute_349, permute_350, alias_27, permute_352, permute_353, permute_356, div_80, permute_361, permute_365, div_81, permute_370, permute_375, permute_376, alias_28, permute_378, permute_379, permute_382, div_82, permute_387, permute_391, div_83, permute_396, permute_401, permute_402, alias_29, permute_404, permute_405, permute_408, div_84, permute_413, permute_417, div_85, permute_422, permute_427, permute_428, alias_30, permute_430, permute_431, permute_434, div_86, permute_439, permute_443, div_87, permute_448, permute_453, permute_454, alias_31, permute_456, permute_457, permute_460, div_88, permute_465, permute_469, div_89, permute_474, permute_479, permute_480, alias_32, permute_482, permute_483, permute_486, div_90, permute_491, permute_495, div_91, permute_500, permute_505, permute_506, alias_33, permute_508, permute_509, permute_512, div_92, permute_517, permute_521, div_93, permute_526, permute_531, permute_532, alias_34, permute_534, permute_535, permute_538, div_94, permute_543, permute_547, div_95, permute_552, permute_557, permute_558, alias_35, permute_560, permute_561, permute_564, div_96, permute_569, permute_573, div_97, permute_578, permute_583, permute_584, alias_36, permute_586, permute_587, permute_590, div_98, permute_595, permute_599, div_99, permute_604, permute_609, permute_610, alias_37, permute_612, permute_613, permute_616, div_100, permute_621, permute_625, div_101, permute_630, permute_635, permute_636, alias_38, permute_638, permute_639, permute_642, div_102, permute_647, permute_651, div_103, permute_656, permute_661, permute_662, alias_39, permute_664, permute_665, permute_668, div_104, permute_673, permute_677, div_105, permute_682, permute_687, permute_688, alias_40, permute_690, permute_691, permute_694, div_106, permute_699, permute_703, div_107, permute_708, permute_713, permute_714, alias_41, permute_716, permute_717, permute_720, div_108, permute_725, permute_729, div_109, permute_734, permute_739, permute_740, alias_42, permute_742, permute_743, permute_746, div_110, permute_751, permute_755, div_111, permute_760, permute_765, permute_766, alias_43, permute_768, permute_769, permute_772, permute_779, div_113, permute_782, permute_786, div_114, permute_791, permute_796, permute_797, alias_44, permute_799, permute_800, permute_803, div_115, permute_808, permute_812, div_116, permute_817, permute_822, permute_823, alias_45, permute_825, permute_826, permute_829, permute_836, div_118, permute_839, permute_843, div_119, permute_848, permute_853, permute_854, alias_46, permute_856, permute_857, permute_860, div_120, permute_865, permute_869, div_121, permute_874, permute_879, permute_880, alias_47, permute_882, permute_883, permute_886, div_122, div_123, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('swin_base_patch4_window7_224', benchmark_compiled_module)
