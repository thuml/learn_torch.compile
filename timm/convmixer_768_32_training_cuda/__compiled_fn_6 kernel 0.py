
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


# kernel path: /tmp/torchinductor_youkaichao/rn/crn2btqha7axs4zejiq75bhxjodjsa6d7sl43t5hj7xu4ofscfeb.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward]

triton_red_fused_div_native_batch_norm_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_batch_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 1024.0
        tmp2 = tmp0 / tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fq/cfquaohqzuzrmiaybl7tgxze5i4jrqydrxwybqsjkcycw7rrqogi.py
# Source Nodes: [l__mod___blocks_31_2], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.relu]
# l__mod___blocks_31_2 => relu_64
triton_red_fused_div_native_batch_norm_backward_relu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_batch_norm_backward_relu_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*((r2 + (128*x1)) // 1024))), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 1024.0
        tmp2 = tmp0 / tmp1
        tmp4 = triton_helpers.maximum(0, tmp3)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp2 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bz/cbzri4tj2rurwg3lgqn6sdxu3bua5uoqlsot2upqaocr5f4nvbmn.py
# Source Nodes: [l__mod___blocks_31_2], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.relu]
# l__mod___blocks_31_2 => relu_64
triton_per_fused_div_native_batch_norm_backward_relu_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_relu_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t3/ct3yae77lspxj5yvjkho7mc2wiwdi7uhytekvzraiaaseslfz7xe.py
# Source Nodes: [l__mod___blocks_31_2], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
# l__mod___blocks_31_2 => relu_64
triton_poi_fused_div_native_batch_norm_backward_relu_threshold_backward_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_native_batch_norm_backward_relu_threshold_backward_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 768
    x2 = (xindex // 786432)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp4 = tl.load(in_ptr1 + (x0 + (768*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tmp5 = 1024.0
    tmp6 = tmp4 / tmp5
    tmp8 = tmp1 - tmp7
    tmp10 = 0.0001220703125
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp6 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tl.where(tmp3, tmp2, tmp22)
    tl.store(out_ptr0 + (x3), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hc/chcgusnxuqcrix76pf7a6ax4xhxp3cmmtoq55blei72l3bb2uaw5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
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
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sy/csy7aiq5pk5h4tzljuwfx4sntswpvpn7jnnre2hcnsx6qkgw5kjn.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 64
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


# kernel path: /tmp/torchinductor_youkaichao/ny/cnyzud6q6kzp4flzxpft2jcvg3tm73cpyp5h5fidmoduaqrxzrau.py
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
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
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
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (786432*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nz/cnzbxuncq2trjxe4ergtbspg6lxts2mwgrnvq6xp27hxiitve2g5.py
# Source Nodes: [getattr_getattr_l__mod___blocks___31_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
# getattr_getattr_l__mod___blocks___31_____0___fn_1 => relu_63
triton_red_fused_native_batch_norm_backward_relu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_relu_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (786432*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (768*r2) + (98304*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = triton_helpers.maximum(0, tmp1)
        tmp4 = tmp2 - tmp3
        tmp5 = tmp0 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ov/covc5z4bsxv7lc4wj3hm3ch5nocentjhchspsztov3zky6hhhby4.py
# Source Nodes: [getattr_getattr_l__mod___blocks___31_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
# getattr_getattr_l__mod___blocks___31_____0___fn_1 => relu_63
triton_per_fused_native_batch_norm_backward_relu_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_relu_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
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


# kernel path: /tmp/torchinductor_youkaichao/5l/c5lda7uyfyxt2jam346ds5b75ru3pqjgttgl4bsjm3b3xp2bzez4.py
# Source Nodes: [getattr_getattr_l__mod___blocks___31_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
# getattr_getattr_l__mod___blocks___31_____0___fn_1 => relu_63
triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x2 + (768*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (y0 + (1024*x2) + (786432*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tmp6 = tmp1 - tmp5
    tmp8 = 0.0001220703125
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp4 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tmp21 = tl.where(tmp3, tmp2, tmp20)
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zm/czmy4segmoq3slnvzj7yhrdx7hkzaem3gfusd7dkimqxrwliqoq7.py
# Source Nodes: [l__mod___blocks_30_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
# l__mod___blocks_30_2 => relu_62
triton_red_fused_add_native_batch_norm_backward_relu_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_relu_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (786432*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0) + (786432*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (768*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp7 = triton_helpers.maximum(0, tmp6)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp2 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp12, xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tmp12 * tmp14
    tl.store(out_ptr2 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u6/cu6cn36yyskxa473gyks6hdzh73lassu4h5ohkfskzuxsuyz65tr.py
# Source Nodes: [l__mod___blocks_30_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
# l__mod___blocks_30_2 => relu_62
triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (768*x2) + (786432*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tmp6 = tmp4 + tmp5
    tmp8 = tmp1 - tmp7
    tmp10 = 0.0001220703125
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp6 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tl.where(tmp3, tmp2, tmp22)
    tl.store(out_ptr0 + (y0 + (768*x2) + (786432*y1)), tmp23, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_191, primals_193, primals_195, primals_197, primals_199, primals_201, primals_203, primals_205, primals_207, primals_209, primals_211, primals_213, primals_215, primals_217, primals_219, primals_221, primals_223, primals_225, primals_227, primals_229, primals_231, primals_233, primals_235, primals_237, primals_239, primals_241, primals_243, primals_245, primals_247, primals_249, primals_251, primals_253, primals_255, primals_257, primals_259, primals_458, convolution, squeeze_1, add_4, convolution_1, squeeze_4, add_10, convolution_2, squeeze_7, add_15, convolution_3, squeeze_10, add_21, convolution_4, squeeze_13, add_26, convolution_5, squeeze_16, add_32, convolution_6, squeeze_19, add_37, convolution_7, squeeze_22, add_43, convolution_8, squeeze_25, add_48, convolution_9, squeeze_28, add_54, convolution_10, squeeze_31, add_59, convolution_11, squeeze_34, add_65, convolution_12, squeeze_37, add_70, convolution_13, squeeze_40, add_76, convolution_14, squeeze_43, add_81, convolution_15, squeeze_46, add_87, convolution_16, squeeze_49, add_92, convolution_17, squeeze_52, add_98, convolution_18, squeeze_55, add_103, convolution_19, squeeze_58, add_109, convolution_20, squeeze_61, add_114, convolution_21, squeeze_64, add_120, convolution_22, squeeze_67, add_125, convolution_23, squeeze_70, add_131, convolution_24, squeeze_73, add_136, convolution_25, squeeze_76, add_142, convolution_26, squeeze_79, add_147, convolution_27, squeeze_82, add_153, convolution_28, squeeze_85, add_158, convolution_29, squeeze_88, add_164, convolution_30, squeeze_91, add_169, convolution_31, squeeze_94, add_175, convolution_32, squeeze_97, add_180, convolution_33, squeeze_100, add_186, convolution_34, squeeze_103, add_191, convolution_35, squeeze_106, add_197, convolution_36, squeeze_109, add_202, convolution_37, squeeze_112, add_208, convolution_38, squeeze_115, add_213, convolution_39, squeeze_118, add_219, convolution_40, squeeze_121, add_224, convolution_41, squeeze_124, add_230, convolution_42, squeeze_127, add_235, convolution_43, squeeze_130, add_241, convolution_44, squeeze_133, add_246, convolution_45, squeeze_136, add_252, convolution_46, squeeze_139, add_257, convolution_47, squeeze_142, add_263, convolution_48, squeeze_145, add_268, convolution_49, squeeze_148, add_274, convolution_50, squeeze_151, add_279, convolution_51, squeeze_154, add_285, convolution_52, squeeze_157, add_290, convolution_53, squeeze_160, add_296, convolution_54, squeeze_163, add_301, convolution_55, squeeze_166, add_307, convolution_56, squeeze_169, add_312, convolution_57, squeeze_172, add_318, convolution_58, squeeze_175, add_323, convolution_59, squeeze_178, add_329, convolution_60, squeeze_181, add_334, convolution_61, squeeze_184, add_340, convolution_62, squeeze_187, add_345, convolution_63, squeeze_190, add_351, convolution_64, squeeze_193, clone, permute_1, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, unsqueeze_850, unsqueeze_862, unsqueeze_874, unsqueeze_886, unsqueeze_898, unsqueeze_910, unsqueeze_922, unsqueeze_934, unsqueeze_946, unsqueeze_958, unsqueeze_970, unsqueeze_982, unsqueeze_994, unsqueeze_1006, unsqueeze_1018, unsqueeze_1030, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (768, 3, 7, 7), (147, 1, 21, 3))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_5, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_9, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_13, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_17, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_21, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_25, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_29, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_33, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_37, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_41, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_45, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_49, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_53, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_57, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_61, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_65, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_69, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_73, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_77, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_81, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_85, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_89, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_91, (768, ), (1, ))
    assert_size_stride(primals_93, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_97, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_101, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_105, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_109, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_113, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_115, (768, ), (1, ))
    assert_size_stride(primals_117, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_121, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_125, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_129, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_131, (768, ), (1, ))
    assert_size_stride(primals_133, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_137, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_141, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_145, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_149, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_151, (768, ), (1, ))
    assert_size_stride(primals_153, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_155, (768, ), (1, ))
    assert_size_stride(primals_157, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_159, (768, ), (1, ))
    assert_size_stride(primals_161, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_163, (768, ), (1, ))
    assert_size_stride(primals_165, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_167, (768, ), (1, ))
    assert_size_stride(primals_169, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_171, (768, ), (1, ))
    assert_size_stride(primals_173, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_175, (768, ), (1, ))
    assert_size_stride(primals_177, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_179, (768, ), (1, ))
    assert_size_stride(primals_181, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_183, (768, ), (1, ))
    assert_size_stride(primals_185, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_187, (768, ), (1, ))
    assert_size_stride(primals_189, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(primals_193, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_195, (768, ), (1, ))
    assert_size_stride(primals_197, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_199, (768, ), (1, ))
    assert_size_stride(primals_201, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_203, (768, ), (1, ))
    assert_size_stride(primals_205, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_207, (768, ), (1, ))
    assert_size_stride(primals_209, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_211, (768, ), (1, ))
    assert_size_stride(primals_213, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_215, (768, ), (1, ))
    assert_size_stride(primals_217, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_219, (768, ), (1, ))
    assert_size_stride(primals_221, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_223, (768, ), (1, ))
    assert_size_stride(primals_225, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_227, (768, ), (1, ))
    assert_size_stride(primals_229, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_231, (768, ), (1, ))
    assert_size_stride(primals_233, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_235, (768, ), (1, ))
    assert_size_stride(primals_237, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_239, (768, ), (1, ))
    assert_size_stride(primals_241, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_243, (768, ), (1, ))
    assert_size_stride(primals_245, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_247, (768, ), (1, ))
    assert_size_stride(primals_249, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_251, (768, ), (1, ))
    assert_size_stride(primals_253, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_255, (768, ), (1, ))
    assert_size_stride(primals_257, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_259, (768, ), (1, ))
    assert_size_stride(primals_458, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_1, (768, ), (1, ))
    assert_size_stride(add_4, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_1, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_4, (768, ), (1, ))
    assert_size_stride(add_10, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_2, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_7, (768, ), (1, ))
    assert_size_stride(add_15, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_3, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_10, (768, ), (1, ))
    assert_size_stride(add_21, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_4, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_13, (768, ), (1, ))
    assert_size_stride(add_26, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_5, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_16, (768, ), (1, ))
    assert_size_stride(add_32, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_6, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_19, (768, ), (1, ))
    assert_size_stride(add_37, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_7, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_22, (768, ), (1, ))
    assert_size_stride(add_43, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_8, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_25, (768, ), (1, ))
    assert_size_stride(add_48, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_9, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_28, (768, ), (1, ))
    assert_size_stride(add_54, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_10, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_31, (768, ), (1, ))
    assert_size_stride(add_59, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_11, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_34, (768, ), (1, ))
    assert_size_stride(add_65, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_12, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_37, (768, ), (1, ))
    assert_size_stride(add_70, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_13, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_40, (768, ), (1, ))
    assert_size_stride(add_76, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_14, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_43, (768, ), (1, ))
    assert_size_stride(add_81, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_15, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_46, (768, ), (1, ))
    assert_size_stride(add_87, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_16, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_49, (768, ), (1, ))
    assert_size_stride(add_92, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_17, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_52, (768, ), (1, ))
    assert_size_stride(add_98, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_18, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_55, (768, ), (1, ))
    assert_size_stride(add_103, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_19, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_58, (768, ), (1, ))
    assert_size_stride(add_109, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_20, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_61, (768, ), (1, ))
    assert_size_stride(add_114, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_21, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_64, (768, ), (1, ))
    assert_size_stride(add_120, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_22, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_67, (768, ), (1, ))
    assert_size_stride(add_125, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_23, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_70, (768, ), (1, ))
    assert_size_stride(add_131, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_24, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_73, (768, ), (1, ))
    assert_size_stride(add_136, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_25, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_76, (768, ), (1, ))
    assert_size_stride(add_142, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_26, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_79, (768, ), (1, ))
    assert_size_stride(add_147, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_27, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_82, (768, ), (1, ))
    assert_size_stride(add_153, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_28, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_85, (768, ), (1, ))
    assert_size_stride(add_158, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_29, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_88, (768, ), (1, ))
    assert_size_stride(add_164, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_30, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_91, (768, ), (1, ))
    assert_size_stride(add_169, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_31, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_94, (768, ), (1, ))
    assert_size_stride(add_175, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_32, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_97, (768, ), (1, ))
    assert_size_stride(add_180, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_33, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_100, (768, ), (1, ))
    assert_size_stride(add_186, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_34, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_103, (768, ), (1, ))
    assert_size_stride(add_191, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_35, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_106, (768, ), (1, ))
    assert_size_stride(add_197, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_36, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_109, (768, ), (1, ))
    assert_size_stride(add_202, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_37, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_112, (768, ), (1, ))
    assert_size_stride(add_208, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_38, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_115, (768, ), (1, ))
    assert_size_stride(add_213, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_39, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_118, (768, ), (1, ))
    assert_size_stride(add_219, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_40, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_121, (768, ), (1, ))
    assert_size_stride(add_224, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_41, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_124, (768, ), (1, ))
    assert_size_stride(add_230, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_42, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_127, (768, ), (1, ))
    assert_size_stride(add_235, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_43, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_130, (768, ), (1, ))
    assert_size_stride(add_241, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_44, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_133, (768, ), (1, ))
    assert_size_stride(add_246, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_45, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_136, (768, ), (1, ))
    assert_size_stride(add_252, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_46, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_139, (768, ), (1, ))
    assert_size_stride(add_257, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_47, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_142, (768, ), (1, ))
    assert_size_stride(add_263, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_48, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_145, (768, ), (1, ))
    assert_size_stride(add_268, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_49, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_148, (768, ), (1, ))
    assert_size_stride(add_274, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_50, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_151, (768, ), (1, ))
    assert_size_stride(add_279, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_51, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_154, (768, ), (1, ))
    assert_size_stride(add_285, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_52, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_157, (768, ), (1, ))
    assert_size_stride(add_290, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_53, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_160, (768, ), (1, ))
    assert_size_stride(add_296, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_54, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_163, (768, ), (1, ))
    assert_size_stride(add_301, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_55, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_166, (768, ), (1, ))
    assert_size_stride(add_307, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_56, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_169, (768, ), (1, ))
    assert_size_stride(add_312, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_57, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_172, (768, ), (1, ))
    assert_size_stride(add_318, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_58, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_175, (768, ), (1, ))
    assert_size_stride(add_323, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_59, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_178, (768, ), (1, ))
    assert_size_stride(add_329, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_60, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_181, (768, ), (1, ))
    assert_size_stride(add_334, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_61, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_184, (768, ), (1, ))
    assert_size_stride(add_340, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_62, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_187, (768, ), (1, ))
    assert_size_stride(add_345, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_63, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_190, (768, ), (1, ))
    assert_size_stride(add_351, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_64, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_193, (768, ), (1, ))
    assert_size_stride(clone, (8, 768), (768, 1))
    assert_size_stride(permute_1, (1000, 768), (768, 1))
    assert_size_stride(unsqueeze_262, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_274, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_286, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_298, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_310, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_322, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_334, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_346, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_358, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_370, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_382, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_394, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_406, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_418, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_430, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_442, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_454, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_466, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_478, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_490, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_502, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_514, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_526, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_538, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_550, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_562, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_574, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_586, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_598, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_610, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_622, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_634, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_646, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_658, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_670, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_682, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_694, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_706, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_718, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_730, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_742, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_754, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_766, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_778, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_790, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_802, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_814, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_826, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_838, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_850, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_862, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_874, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_886, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_898, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_910, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_922, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_934, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_946, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_958, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_970, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_982, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_994, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_1006, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_1018, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_1030, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone, out=buf1)
        del clone
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward]
        triton_red_fused_div_native_batch_norm_backward_1.run(buf0, buf3, 768, 8192, grid=grid(768), stream=stream0)
        buf4 = empty_strided((768, 64), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_31_2], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_div_native_batch_norm_backward_relu_2.run(buf0, convolution_64, unsqueeze_262, buf4, 49152, 128, grid=grid(49152), stream=stream0)
        buf5 = empty((768, ), device='cuda', dtype=torch.float32)
        buf6 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_31_2], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_div_native_batch_norm_backward_relu_3.run(buf4, squeeze_193, buf5, buf6, 768, 64, grid=grid(768), stream=stream0)
        buf7 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_31_2], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_div_native_batch_norm_backward_relu_threshold_backward_4.run(convolution_64, buf0, unsqueeze_262, buf5, squeeze_193, buf3, primals_259, buf7, 6291456, grid=grid(6291456), stream=stream0)
        del buf0
        del convolution_64
        del primals_259
        del squeeze_193
        del unsqueeze_262
        buf8 = buf4; del buf4  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf7, buf8, 49152, 128, grid=grid(49152), stream=stream0)
        buf9 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf8, buf9, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf10 = aten.convolution_backward(buf7, add_351, primals_257, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_351
        del primals_257
        buf11 = buf10[0]
        buf12 = buf10[1]
        del buf10
        buf13 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf11, buf13, 768, 8192, grid=grid(768), stream=stream0)
        buf14 = reinterpret_tensor(buf8, (768, 64), (64, 1), 0); del buf8  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___31_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf11, convolution_63, unsqueeze_274, buf14, 49152, 128, grid=grid(49152), stream=stream0)
        buf15 = empty((768, ), device='cuda', dtype=torch.float32)
        buf16 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___31_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf14, squeeze_190, buf15, buf16, 768, 64, grid=grid(768), stream=stream0)
        buf17 = buf7; del buf7  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___31_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_63, buf11, unsqueeze_274, buf15, squeeze_190, buf13, primals_255, buf17, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_63
        del primals_255
        del squeeze_190
        del unsqueeze_274
        buf18 = reinterpret_tensor(buf14, (768, 64), (1, 768), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf17, buf18, 49152, 128, grid=grid(49152), stream=stream0)
        buf19 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf18, buf19, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf20 = aten.convolution_backward(buf17, add_345, primals_253, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_345
        del primals_253
        buf21 = buf20[0]
        buf22 = buf20[1]
        del buf20
        buf23 = empty((768, ), device='cuda', dtype=torch.float32)
        buf24 = empty((768, ), device='cuda', dtype=torch.float32)
        buf25 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_30_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf11, buf21, convolution_62, unsqueeze_286, squeeze_187, buf23, buf24, buf25, 768, 8192, grid=grid(768), stream=stream0)
        buf26 = buf17; del buf17  # reuse
        # Source Nodes: [l__mod___blocks_30_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_62, buf11, buf21, unsqueeze_286, buf24, squeeze_187, buf23, primals_251, buf26, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf11
        del buf21
        del convolution_62
        del primals_251
        del squeeze_187
        del unsqueeze_286
        buf27 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf26, buf27, 49152, 128, grid=grid(49152), stream=stream0)
        buf28 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf27, buf28, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf29 = aten.convolution_backward(buf26, add_340, primals_249, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_340
        del primals_249
        buf30 = buf29[0]
        buf31 = buf29[1]
        del buf29
        buf32 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf30, buf32, 768, 8192, grid=grid(768), stream=stream0)
        buf33 = reinterpret_tensor(buf27, (768, 64), (64, 1), 0); del buf27  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___30_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf30, convolution_61, unsqueeze_298, buf33, 49152, 128, grid=grid(49152), stream=stream0)
        buf34 = empty((768, ), device='cuda', dtype=torch.float32)
        buf35 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___30_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf33, squeeze_184, buf34, buf35, 768, 64, grid=grid(768), stream=stream0)
        buf36 = buf26; del buf26  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___30_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_61, buf30, unsqueeze_298, buf34, squeeze_184, buf32, primals_247, buf36, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_61
        del primals_247
        del squeeze_184
        del unsqueeze_298
        buf37 = reinterpret_tensor(buf33, (768, 64), (1, 768), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf36, buf37, 49152, 128, grid=grid(49152), stream=stream0)
        buf38 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf37, buf38, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf39 = aten.convolution_backward(buf36, add_334, primals_245, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_334
        del primals_245
        buf40 = buf39[0]
        buf41 = buf39[1]
        del buf39
        buf42 = empty((768, ), device='cuda', dtype=torch.float32)
        buf43 = empty((768, ), device='cuda', dtype=torch.float32)
        buf44 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_29_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf30, buf40, convolution_60, unsqueeze_310, squeeze_181, buf42, buf43, buf44, 768, 8192, grid=grid(768), stream=stream0)
        buf45 = buf36; del buf36  # reuse
        # Source Nodes: [l__mod___blocks_29_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_60, buf30, buf40, unsqueeze_310, buf43, squeeze_181, buf42, primals_243, buf45, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf30
        del buf40
        del convolution_60
        del primals_243
        del squeeze_181
        del unsqueeze_310
        buf46 = buf37; del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf45, buf46, 49152, 128, grid=grid(49152), stream=stream0)
        buf47 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf46, buf47, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf48 = aten.convolution_backward(buf45, add_329, primals_241, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_329
        del primals_241
        buf49 = buf48[0]
        buf50 = buf48[1]
        del buf48
        buf51 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf49, buf51, 768, 8192, grid=grid(768), stream=stream0)
        buf52 = reinterpret_tensor(buf46, (768, 64), (64, 1), 0); del buf46  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___29_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf49, convolution_59, unsqueeze_322, buf52, 49152, 128, grid=grid(49152), stream=stream0)
        buf53 = empty((768, ), device='cuda', dtype=torch.float32)
        buf54 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___29_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf52, squeeze_178, buf53, buf54, 768, 64, grid=grid(768), stream=stream0)
        buf55 = buf45; del buf45  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___29_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_59, buf49, unsqueeze_322, buf53, squeeze_178, buf51, primals_239, buf55, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_59
        del primals_239
        del squeeze_178
        del unsqueeze_322
        buf56 = reinterpret_tensor(buf52, (768, 64), (1, 768), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf55, buf56, 49152, 128, grid=grid(49152), stream=stream0)
        buf57 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf56, buf57, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf58 = aten.convolution_backward(buf55, add_323, primals_237, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_323
        del primals_237
        buf59 = buf58[0]
        buf60 = buf58[1]
        del buf58
        buf61 = empty((768, ), device='cuda', dtype=torch.float32)
        buf62 = empty((768, ), device='cuda', dtype=torch.float32)
        buf63 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_28_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf49, buf59, convolution_58, unsqueeze_334, squeeze_175, buf61, buf62, buf63, 768, 8192, grid=grid(768), stream=stream0)
        buf64 = buf55; del buf55  # reuse
        # Source Nodes: [l__mod___blocks_28_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_58, buf49, buf59, unsqueeze_334, buf62, squeeze_175, buf61, primals_235, buf64, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf49
        del buf59
        del convolution_58
        del primals_235
        del squeeze_175
        del unsqueeze_334
        buf65 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf64, buf65, 49152, 128, grid=grid(49152), stream=stream0)
        buf66 = buf62; del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf65, buf66, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf67 = aten.convolution_backward(buf64, add_318, primals_233, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_318
        del primals_233
        buf68 = buf67[0]
        buf69 = buf67[1]
        del buf67
        buf70 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf68, buf70, 768, 8192, grid=grid(768), stream=stream0)
        buf71 = reinterpret_tensor(buf65, (768, 64), (64, 1), 0); del buf65  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___28_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf68, convolution_57, unsqueeze_346, buf71, 49152, 128, grid=grid(49152), stream=stream0)
        buf72 = empty((768, ), device='cuda', dtype=torch.float32)
        buf73 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___28_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf71, squeeze_172, buf72, buf73, 768, 64, grid=grid(768), stream=stream0)
        buf74 = buf64; del buf64  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___28_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_57, buf68, unsqueeze_346, buf72, squeeze_172, buf70, primals_231, buf74, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_57
        del primals_231
        del squeeze_172
        del unsqueeze_346
        buf75 = reinterpret_tensor(buf71, (768, 64), (1, 768), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf74, buf75, 49152, 128, grid=grid(49152), stream=stream0)
        buf76 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf75, buf76, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf77 = aten.convolution_backward(buf74, add_312, primals_229, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_312
        del primals_229
        buf78 = buf77[0]
        buf79 = buf77[1]
        del buf77
        buf80 = empty((768, ), device='cuda', dtype=torch.float32)
        buf81 = empty((768, ), device='cuda', dtype=torch.float32)
        buf82 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_27_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf68, buf78, convolution_56, unsqueeze_358, squeeze_169, buf80, buf81, buf82, 768, 8192, grid=grid(768), stream=stream0)
        buf83 = buf74; del buf74  # reuse
        # Source Nodes: [l__mod___blocks_27_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_56, buf68, buf78, unsqueeze_358, buf81, squeeze_169, buf80, primals_227, buf83, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf68
        del buf78
        del convolution_56
        del primals_227
        del squeeze_169
        del unsqueeze_358
        buf84 = buf75; del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf83, buf84, 49152, 128, grid=grid(49152), stream=stream0)
        buf85 = buf81; del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf84, buf85, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf86 = aten.convolution_backward(buf83, add_307, primals_225, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_307
        del primals_225
        buf87 = buf86[0]
        buf88 = buf86[1]
        del buf86
        buf89 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf87, buf89, 768, 8192, grid=grid(768), stream=stream0)
        buf90 = reinterpret_tensor(buf84, (768, 64), (64, 1), 0); del buf84  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___27_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf87, convolution_55, unsqueeze_370, buf90, 49152, 128, grid=grid(49152), stream=stream0)
        buf91 = empty((768, ), device='cuda', dtype=torch.float32)
        buf92 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___27_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf90, squeeze_166, buf91, buf92, 768, 64, grid=grid(768), stream=stream0)
        buf93 = buf83; del buf83  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___27_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_55, buf87, unsqueeze_370, buf91, squeeze_166, buf89, primals_223, buf93, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_55
        del primals_223
        del squeeze_166
        del unsqueeze_370
        buf94 = reinterpret_tensor(buf90, (768, 64), (1, 768), 0); del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf93, buf94, 49152, 128, grid=grid(49152), stream=stream0)
        buf95 = buf91; del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf94, buf95, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf96 = aten.convolution_backward(buf93, add_301, primals_221, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_301
        del primals_221
        buf97 = buf96[0]
        buf98 = buf96[1]
        del buf96
        buf99 = empty((768, ), device='cuda', dtype=torch.float32)
        buf100 = empty((768, ), device='cuda', dtype=torch.float32)
        buf101 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_26_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf87, buf97, convolution_54, unsqueeze_382, squeeze_163, buf99, buf100, buf101, 768, 8192, grid=grid(768), stream=stream0)
        buf102 = buf93; del buf93  # reuse
        # Source Nodes: [l__mod___blocks_26_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_54, buf87, buf97, unsqueeze_382, buf100, squeeze_163, buf99, primals_219, buf102, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf87
        del buf97
        del convolution_54
        del primals_219
        del squeeze_163
        del unsqueeze_382
        buf103 = buf94; del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf102, buf103, 49152, 128, grid=grid(49152), stream=stream0)
        buf104 = buf100; del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf103, buf104, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf105 = aten.convolution_backward(buf102, add_296, primals_217, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_296
        del primals_217
        buf106 = buf105[0]
        buf107 = buf105[1]
        del buf105
        buf108 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf106, buf108, 768, 8192, grid=grid(768), stream=stream0)
        buf109 = reinterpret_tensor(buf103, (768, 64), (64, 1), 0); del buf103  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___26_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf106, convolution_53, unsqueeze_394, buf109, 49152, 128, grid=grid(49152), stream=stream0)
        buf110 = empty((768, ), device='cuda', dtype=torch.float32)
        buf111 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___26_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf109, squeeze_160, buf110, buf111, 768, 64, grid=grid(768), stream=stream0)
        buf112 = buf102; del buf102  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___26_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_53, buf106, unsqueeze_394, buf110, squeeze_160, buf108, primals_215, buf112, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_53
        del primals_215
        del squeeze_160
        del unsqueeze_394
        buf113 = reinterpret_tensor(buf109, (768, 64), (1, 768), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf112, buf113, 49152, 128, grid=grid(49152), stream=stream0)
        buf114 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf113, buf114, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf115 = aten.convolution_backward(buf112, add_290, primals_213, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_290
        del primals_213
        buf116 = buf115[0]
        buf117 = buf115[1]
        del buf115
        buf118 = empty((768, ), device='cuda', dtype=torch.float32)
        buf119 = empty((768, ), device='cuda', dtype=torch.float32)
        buf120 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_25_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf106, buf116, convolution_52, unsqueeze_406, squeeze_157, buf118, buf119, buf120, 768, 8192, grid=grid(768), stream=stream0)
        buf121 = buf112; del buf112  # reuse
        # Source Nodes: [l__mod___blocks_25_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_52, buf106, buf116, unsqueeze_406, buf119, squeeze_157, buf118, primals_211, buf121, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf106
        del buf116
        del convolution_52
        del primals_211
        del squeeze_157
        del unsqueeze_406
        buf122 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf121, buf122, 49152, 128, grid=grid(49152), stream=stream0)
        buf123 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf122, buf123, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf124 = aten.convolution_backward(buf121, add_285, primals_209, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_285
        del primals_209
        buf125 = buf124[0]
        buf126 = buf124[1]
        del buf124
        buf127 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf125, buf127, 768, 8192, grid=grid(768), stream=stream0)
        buf128 = reinterpret_tensor(buf122, (768, 64), (64, 1), 0); del buf122  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___25_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf125, convolution_51, unsqueeze_418, buf128, 49152, 128, grid=grid(49152), stream=stream0)
        buf129 = empty((768, ), device='cuda', dtype=torch.float32)
        buf130 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___25_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf128, squeeze_154, buf129, buf130, 768, 64, grid=grid(768), stream=stream0)
        buf131 = buf121; del buf121  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___25_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_51, buf125, unsqueeze_418, buf129, squeeze_154, buf127, primals_207, buf131, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_51
        del primals_207
        del squeeze_154
        del unsqueeze_418
        buf132 = reinterpret_tensor(buf128, (768, 64), (1, 768), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf131, buf132, 49152, 128, grid=grid(49152), stream=stream0)
        buf133 = buf129; del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf132, buf133, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf134 = aten.convolution_backward(buf131, add_279, primals_205, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_279
        del primals_205
        buf135 = buf134[0]
        buf136 = buf134[1]
        del buf134
        buf137 = empty((768, ), device='cuda', dtype=torch.float32)
        buf138 = empty((768, ), device='cuda', dtype=torch.float32)
        buf139 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_24_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf125, buf135, convolution_50, unsqueeze_430, squeeze_151, buf137, buf138, buf139, 768, 8192, grid=grid(768), stream=stream0)
        buf140 = buf131; del buf131  # reuse
        # Source Nodes: [l__mod___blocks_24_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_50, buf125, buf135, unsqueeze_430, buf138, squeeze_151, buf137, primals_203, buf140, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf125
        del buf135
        del convolution_50
        del primals_203
        del squeeze_151
        del unsqueeze_430
        buf141 = buf132; del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf140, buf141, 49152, 128, grid=grid(49152), stream=stream0)
        buf142 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf141, buf142, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf143 = aten.convolution_backward(buf140, add_274, primals_201, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_274
        del primals_201
        buf144 = buf143[0]
        buf145 = buf143[1]
        del buf143
        buf146 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf144, buf146, 768, 8192, grid=grid(768), stream=stream0)
        buf147 = reinterpret_tensor(buf141, (768, 64), (64, 1), 0); del buf141  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___24_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf144, convolution_49, unsqueeze_442, buf147, 49152, 128, grid=grid(49152), stream=stream0)
        buf148 = empty((768, ), device='cuda', dtype=torch.float32)
        buf149 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___24_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf147, squeeze_148, buf148, buf149, 768, 64, grid=grid(768), stream=stream0)
        buf150 = buf140; del buf140  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___24_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_49, buf144, unsqueeze_442, buf148, squeeze_148, buf146, primals_199, buf150, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_49
        del primals_199
        del squeeze_148
        del unsqueeze_442
        buf151 = reinterpret_tensor(buf147, (768, 64), (1, 768), 0); del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf150, buf151, 49152, 128, grid=grid(49152), stream=stream0)
        buf152 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf151, buf152, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf153 = aten.convolution_backward(buf150, add_268, primals_197, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_268
        del primals_197
        buf154 = buf153[0]
        buf155 = buf153[1]
        del buf153
        buf156 = empty((768, ), device='cuda', dtype=torch.float32)
        buf157 = empty((768, ), device='cuda', dtype=torch.float32)
        buf158 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_23_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf144, buf154, convolution_48, unsqueeze_454, squeeze_145, buf156, buf157, buf158, 768, 8192, grid=grid(768), stream=stream0)
        buf159 = buf150; del buf150  # reuse
        # Source Nodes: [l__mod___blocks_23_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_48, buf144, buf154, unsqueeze_454, buf157, squeeze_145, buf156, primals_195, buf159, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf144
        del buf154
        del convolution_48
        del primals_195
        del squeeze_145
        del unsqueeze_454
        buf160 = buf151; del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf159, buf160, 49152, 128, grid=grid(49152), stream=stream0)
        buf161 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf160, buf161, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf162 = aten.convolution_backward(buf159, add_263, primals_193, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_263
        del primals_193
        buf163 = buf162[0]
        buf164 = buf162[1]
        del buf162
        buf165 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf163, buf165, 768, 8192, grid=grid(768), stream=stream0)
        buf166 = reinterpret_tensor(buf160, (768, 64), (64, 1), 0); del buf160  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___23_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf163, convolution_47, unsqueeze_466, buf166, 49152, 128, grid=grid(49152), stream=stream0)
        buf167 = empty((768, ), device='cuda', dtype=torch.float32)
        buf168 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___23_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf166, squeeze_142, buf167, buf168, 768, 64, grid=grid(768), stream=stream0)
        buf169 = buf159; del buf159  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___23_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_47, buf163, unsqueeze_466, buf167, squeeze_142, buf165, primals_191, buf169, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_47
        del primals_191
        del squeeze_142
        del unsqueeze_466
        buf170 = reinterpret_tensor(buf166, (768, 64), (1, 768), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf169, buf170, 49152, 128, grid=grid(49152), stream=stream0)
        buf171 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf170, buf171, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf172 = aten.convolution_backward(buf169, add_257, primals_189, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_257
        del primals_189
        buf173 = buf172[0]
        buf174 = buf172[1]
        del buf172
        buf175 = empty((768, ), device='cuda', dtype=torch.float32)
        buf176 = empty((768, ), device='cuda', dtype=torch.float32)
        buf177 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_22_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf163, buf173, convolution_46, unsqueeze_478, squeeze_139, buf175, buf176, buf177, 768, 8192, grid=grid(768), stream=stream0)
        buf178 = buf169; del buf169  # reuse
        # Source Nodes: [l__mod___blocks_22_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_46, buf163, buf173, unsqueeze_478, buf176, squeeze_139, buf175, primals_187, buf178, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf163
        del buf173
        del convolution_46
        del primals_187
        del squeeze_139
        del unsqueeze_478
        buf179 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf178, buf179, 49152, 128, grid=grid(49152), stream=stream0)
        buf180 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf179, buf180, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf181 = aten.convolution_backward(buf178, add_252, primals_185, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_252
        del primals_185
        buf182 = buf181[0]
        buf183 = buf181[1]
        del buf181
        buf184 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf182, buf184, 768, 8192, grid=grid(768), stream=stream0)
        buf185 = reinterpret_tensor(buf179, (768, 64), (64, 1), 0); del buf179  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___22_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf182, convolution_45, unsqueeze_490, buf185, 49152, 128, grid=grid(49152), stream=stream0)
        buf186 = empty((768, ), device='cuda', dtype=torch.float32)
        buf187 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___22_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf185, squeeze_136, buf186, buf187, 768, 64, grid=grid(768), stream=stream0)
        buf188 = buf178; del buf178  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___22_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_45, buf182, unsqueeze_490, buf186, squeeze_136, buf184, primals_183, buf188, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_45
        del primals_183
        del squeeze_136
        del unsqueeze_490
        buf189 = reinterpret_tensor(buf185, (768, 64), (1, 768), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf188, buf189, 49152, 128, grid=grid(49152), stream=stream0)
        buf190 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf189, buf190, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf191 = aten.convolution_backward(buf188, add_246, primals_181, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_246
        del primals_181
        buf192 = buf191[0]
        buf193 = buf191[1]
        del buf191
        buf194 = empty((768, ), device='cuda', dtype=torch.float32)
        buf195 = empty((768, ), device='cuda', dtype=torch.float32)
        buf196 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_21_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf182, buf192, convolution_44, unsqueeze_502, squeeze_133, buf194, buf195, buf196, 768, 8192, grid=grid(768), stream=stream0)
        buf197 = buf188; del buf188  # reuse
        # Source Nodes: [l__mod___blocks_21_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_44, buf182, buf192, unsqueeze_502, buf195, squeeze_133, buf194, primals_179, buf197, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf182
        del buf192
        del convolution_44
        del primals_179
        del squeeze_133
        del unsqueeze_502
        buf198 = buf189; del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf197, buf198, 49152, 128, grid=grid(49152), stream=stream0)
        buf199 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf198, buf199, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf200 = aten.convolution_backward(buf197, add_241, primals_177, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_241
        del primals_177
        buf201 = buf200[0]
        buf202 = buf200[1]
        del buf200
        buf203 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf201, buf203, 768, 8192, grid=grid(768), stream=stream0)
        buf204 = reinterpret_tensor(buf198, (768, 64), (64, 1), 0); del buf198  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___21_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf201, convolution_43, unsqueeze_514, buf204, 49152, 128, grid=grid(49152), stream=stream0)
        buf205 = empty((768, ), device='cuda', dtype=torch.float32)
        buf206 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___21_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf204, squeeze_130, buf205, buf206, 768, 64, grid=grid(768), stream=stream0)
        buf207 = buf197; del buf197  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___21_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_43, buf201, unsqueeze_514, buf205, squeeze_130, buf203, primals_175, buf207, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_43
        del primals_175
        del squeeze_130
        del unsqueeze_514
        buf208 = reinterpret_tensor(buf204, (768, 64), (1, 768), 0); del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf207, buf208, 49152, 128, grid=grid(49152), stream=stream0)
        buf209 = buf205; del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf208, buf209, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf210 = aten.convolution_backward(buf207, add_235, primals_173, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_235
        del primals_173
        buf211 = buf210[0]
        buf212 = buf210[1]
        del buf210
        buf213 = empty((768, ), device='cuda', dtype=torch.float32)
        buf214 = empty((768, ), device='cuda', dtype=torch.float32)
        buf215 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_20_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf201, buf211, convolution_42, unsqueeze_526, squeeze_127, buf213, buf214, buf215, 768, 8192, grid=grid(768), stream=stream0)
        buf216 = buf207; del buf207  # reuse
        # Source Nodes: [l__mod___blocks_20_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_42, buf201, buf211, unsqueeze_526, buf214, squeeze_127, buf213, primals_171, buf216, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf201
        del buf211
        del convolution_42
        del primals_171
        del squeeze_127
        del unsqueeze_526
        buf217 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf216, buf217, 49152, 128, grid=grid(49152), stream=stream0)
        buf218 = buf214; del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf217, buf218, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf219 = aten.convolution_backward(buf216, add_230, primals_169, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_230
        del primals_169
        buf220 = buf219[0]
        buf221 = buf219[1]
        del buf219
        buf222 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf220, buf222, 768, 8192, grid=grid(768), stream=stream0)
        buf223 = reinterpret_tensor(buf217, (768, 64), (64, 1), 0); del buf217  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___20_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf220, convolution_41, unsqueeze_538, buf223, 49152, 128, grid=grid(49152), stream=stream0)
        buf224 = empty((768, ), device='cuda', dtype=torch.float32)
        buf225 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___20_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf223, squeeze_124, buf224, buf225, 768, 64, grid=grid(768), stream=stream0)
        buf226 = buf216; del buf216  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___20_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_41, buf220, unsqueeze_538, buf224, squeeze_124, buf222, primals_167, buf226, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_41
        del primals_167
        del squeeze_124
        del unsqueeze_538
        buf227 = reinterpret_tensor(buf223, (768, 64), (1, 768), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf226, buf227, 49152, 128, grid=grid(49152), stream=stream0)
        buf228 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf227, buf228, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf229 = aten.convolution_backward(buf226, add_224, primals_165, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_224
        del primals_165
        buf230 = buf229[0]
        buf231 = buf229[1]
        del buf229
        buf232 = empty((768, ), device='cuda', dtype=torch.float32)
        buf233 = empty((768, ), device='cuda', dtype=torch.float32)
        buf234 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_19_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf220, buf230, convolution_40, unsqueeze_550, squeeze_121, buf232, buf233, buf234, 768, 8192, grid=grid(768), stream=stream0)
        buf235 = buf226; del buf226  # reuse
        # Source Nodes: [l__mod___blocks_19_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_40, buf220, buf230, unsqueeze_550, buf233, squeeze_121, buf232, primals_163, buf235, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf220
        del buf230
        del convolution_40
        del primals_163
        del squeeze_121
        del unsqueeze_550
        buf236 = buf227; del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf235, buf236, 49152, 128, grid=grid(49152), stream=stream0)
        buf237 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf236, buf237, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf238 = aten.convolution_backward(buf235, add_219, primals_161, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_219
        del primals_161
        buf239 = buf238[0]
        buf240 = buf238[1]
        del buf238
        buf241 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf239, buf241, 768, 8192, grid=grid(768), stream=stream0)
        buf242 = reinterpret_tensor(buf236, (768, 64), (64, 1), 0); del buf236  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___19_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf239, convolution_39, unsqueeze_562, buf242, 49152, 128, grid=grid(49152), stream=stream0)
        buf243 = empty((768, ), device='cuda', dtype=torch.float32)
        buf244 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___19_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf242, squeeze_118, buf243, buf244, 768, 64, grid=grid(768), stream=stream0)
        buf245 = buf235; del buf235  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___19_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_39, buf239, unsqueeze_562, buf243, squeeze_118, buf241, primals_159, buf245, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_39
        del primals_159
        del squeeze_118
        del unsqueeze_562
        buf246 = reinterpret_tensor(buf242, (768, 64), (1, 768), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf245, buf246, 49152, 128, grid=grid(49152), stream=stream0)
        buf247 = buf243; del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf246, buf247, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf248 = aten.convolution_backward(buf245, add_213, primals_157, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_213
        del primals_157
        buf249 = buf248[0]
        buf250 = buf248[1]
        del buf248
        buf251 = empty((768, ), device='cuda', dtype=torch.float32)
        buf252 = empty((768, ), device='cuda', dtype=torch.float32)
        buf253 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_18_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf239, buf249, convolution_38, unsqueeze_574, squeeze_115, buf251, buf252, buf253, 768, 8192, grid=grid(768), stream=stream0)
        buf254 = buf245; del buf245  # reuse
        # Source Nodes: [l__mod___blocks_18_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_38, buf239, buf249, unsqueeze_574, buf252, squeeze_115, buf251, primals_155, buf254, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf239
        del buf249
        del convolution_38
        del primals_155
        del squeeze_115
        del unsqueeze_574
        buf255 = buf246; del buf246  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf254, buf255, 49152, 128, grid=grid(49152), stream=stream0)
        buf256 = buf252; del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf255, buf256, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf257 = aten.convolution_backward(buf254, add_208, primals_153, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_208
        del primals_153
        buf258 = buf257[0]
        buf259 = buf257[1]
        del buf257
        buf260 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf258, buf260, 768, 8192, grid=grid(768), stream=stream0)
        buf261 = reinterpret_tensor(buf255, (768, 64), (64, 1), 0); del buf255  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___18_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf258, convolution_37, unsqueeze_586, buf261, 49152, 128, grid=grid(49152), stream=stream0)
        buf262 = empty((768, ), device='cuda', dtype=torch.float32)
        buf263 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___18_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf261, squeeze_112, buf262, buf263, 768, 64, grid=grid(768), stream=stream0)
        buf264 = buf254; del buf254  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___18_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_37, buf258, unsqueeze_586, buf262, squeeze_112, buf260, primals_151, buf264, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_37
        del primals_151
        del squeeze_112
        del unsqueeze_586
        buf265 = reinterpret_tensor(buf261, (768, 64), (1, 768), 0); del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf264, buf265, 49152, 128, grid=grid(49152), stream=stream0)
        buf266 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf265, buf266, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf267 = aten.convolution_backward(buf264, add_202, primals_149, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_202
        del primals_149
        buf268 = buf267[0]
        buf269 = buf267[1]
        del buf267
        buf270 = empty((768, ), device='cuda', dtype=torch.float32)
        buf271 = empty((768, ), device='cuda', dtype=torch.float32)
        buf272 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_17_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf258, buf268, convolution_36, unsqueeze_598, squeeze_109, buf270, buf271, buf272, 768, 8192, grid=grid(768), stream=stream0)
        buf273 = buf264; del buf264  # reuse
        # Source Nodes: [l__mod___blocks_17_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_36, buf258, buf268, unsqueeze_598, buf271, squeeze_109, buf270, primals_147, buf273, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf258
        del buf268
        del convolution_36
        del primals_147
        del squeeze_109
        del unsqueeze_598
        buf274 = buf265; del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf273, buf274, 49152, 128, grid=grid(49152), stream=stream0)
        buf275 = buf271; del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf274, buf275, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf276 = aten.convolution_backward(buf273, add_197, primals_145, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_197
        del primals_145
        buf277 = buf276[0]
        buf278 = buf276[1]
        del buf276
        buf279 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf277, buf279, 768, 8192, grid=grid(768), stream=stream0)
        buf280 = reinterpret_tensor(buf274, (768, 64), (64, 1), 0); del buf274  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___17_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf277, convolution_35, unsqueeze_610, buf280, 49152, 128, grid=grid(49152), stream=stream0)
        buf281 = empty((768, ), device='cuda', dtype=torch.float32)
        buf282 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___17_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf280, squeeze_106, buf281, buf282, 768, 64, grid=grid(768), stream=stream0)
        buf283 = buf273; del buf273  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___17_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_35, buf277, unsqueeze_610, buf281, squeeze_106, buf279, primals_143, buf283, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_35
        del primals_143
        del squeeze_106
        del unsqueeze_610
        buf284 = reinterpret_tensor(buf280, (768, 64), (1, 768), 0); del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf283, buf284, 49152, 128, grid=grid(49152), stream=stream0)
        buf285 = buf281; del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf284, buf285, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf286 = aten.convolution_backward(buf283, add_191, primals_141, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_191
        del primals_141
        buf287 = buf286[0]
        buf288 = buf286[1]
        del buf286
        buf289 = empty((768, ), device='cuda', dtype=torch.float32)
        buf290 = empty((768, ), device='cuda', dtype=torch.float32)
        buf291 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_16_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf277, buf287, convolution_34, unsqueeze_622, squeeze_103, buf289, buf290, buf291, 768, 8192, grid=grid(768), stream=stream0)
        buf292 = buf283; del buf283  # reuse
        # Source Nodes: [l__mod___blocks_16_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_34, buf277, buf287, unsqueeze_622, buf290, squeeze_103, buf289, primals_139, buf292, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf277
        del buf287
        del convolution_34
        del primals_139
        del squeeze_103
        del unsqueeze_622
        buf293 = buf284; del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf292, buf293, 49152, 128, grid=grid(49152), stream=stream0)
        buf294 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf293, buf294, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf295 = aten.convolution_backward(buf292, add_186, primals_137, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_186
        del primals_137
        buf296 = buf295[0]
        buf297 = buf295[1]
        del buf295
        buf298 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf296, buf298, 768, 8192, grid=grid(768), stream=stream0)
        buf299 = reinterpret_tensor(buf293, (768, 64), (64, 1), 0); del buf293  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___16_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf296, convolution_33, unsqueeze_634, buf299, 49152, 128, grid=grid(49152), stream=stream0)
        buf300 = empty((768, ), device='cuda', dtype=torch.float32)
        buf301 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___16_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf299, squeeze_100, buf300, buf301, 768, 64, grid=grid(768), stream=stream0)
        buf302 = buf292; del buf292  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___16_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_33, buf296, unsqueeze_634, buf300, squeeze_100, buf298, primals_135, buf302, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_33
        del primals_135
        del squeeze_100
        del unsqueeze_634
        buf303 = reinterpret_tensor(buf299, (768, 64), (1, 768), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf302, buf303, 49152, 128, grid=grid(49152), stream=stream0)
        buf304 = buf300; del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf303, buf304, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf305 = aten.convolution_backward(buf302, add_180, primals_133, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_180
        del primals_133
        buf306 = buf305[0]
        buf307 = buf305[1]
        del buf305
        buf308 = empty((768, ), device='cuda', dtype=torch.float32)
        buf309 = empty((768, ), device='cuda', dtype=torch.float32)
        buf310 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_15_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf296, buf306, convolution_32, unsqueeze_646, squeeze_97, buf308, buf309, buf310, 768, 8192, grid=grid(768), stream=stream0)
        buf311 = buf302; del buf302  # reuse
        # Source Nodes: [l__mod___blocks_15_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_32, buf296, buf306, unsqueeze_646, buf309, squeeze_97, buf308, primals_131, buf311, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf296
        del buf306
        del convolution_32
        del primals_131
        del squeeze_97
        del unsqueeze_646
        buf312 = buf303; del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf311, buf312, 49152, 128, grid=grid(49152), stream=stream0)
        buf313 = buf309; del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf312, buf313, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf314 = aten.convolution_backward(buf311, add_175, primals_129, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_175
        del primals_129
        buf315 = buf314[0]
        buf316 = buf314[1]
        del buf314
        buf317 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf315, buf317, 768, 8192, grid=grid(768), stream=stream0)
        buf318 = reinterpret_tensor(buf312, (768, 64), (64, 1), 0); del buf312  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___15_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf315, convolution_31, unsqueeze_658, buf318, 49152, 128, grid=grid(49152), stream=stream0)
        buf319 = empty((768, ), device='cuda', dtype=torch.float32)
        buf320 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___15_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf318, squeeze_94, buf319, buf320, 768, 64, grid=grid(768), stream=stream0)
        buf321 = buf311; del buf311  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___15_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_31, buf315, unsqueeze_658, buf319, squeeze_94, buf317, primals_127, buf321, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_31
        del primals_127
        del squeeze_94
        del unsqueeze_658
        buf322 = reinterpret_tensor(buf318, (768, 64), (1, 768), 0); del buf318  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf321, buf322, 49152, 128, grid=grid(49152), stream=stream0)
        buf323 = buf319; del buf319  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf322, buf323, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf324 = aten.convolution_backward(buf321, add_169, primals_125, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_169
        del primals_125
        buf325 = buf324[0]
        buf326 = buf324[1]
        del buf324
        buf327 = empty((768, ), device='cuda', dtype=torch.float32)
        buf328 = empty((768, ), device='cuda', dtype=torch.float32)
        buf329 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_14_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf315, buf325, convolution_30, unsqueeze_670, squeeze_91, buf327, buf328, buf329, 768, 8192, grid=grid(768), stream=stream0)
        buf330 = buf321; del buf321  # reuse
        # Source Nodes: [l__mod___blocks_14_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_30, buf315, buf325, unsqueeze_670, buf328, squeeze_91, buf327, primals_123, buf330, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf315
        del buf325
        del convolution_30
        del primals_123
        del squeeze_91
        del unsqueeze_670
        buf331 = buf322; del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf330, buf331, 49152, 128, grid=grid(49152), stream=stream0)
        buf332 = buf328; del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf331, buf332, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf333 = aten.convolution_backward(buf330, add_164, primals_121, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_164
        del primals_121
        buf334 = buf333[0]
        buf335 = buf333[1]
        del buf333
        buf336 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf334, buf336, 768, 8192, grid=grid(768), stream=stream0)
        buf337 = reinterpret_tensor(buf331, (768, 64), (64, 1), 0); del buf331  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___14_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf334, convolution_29, unsqueeze_682, buf337, 49152, 128, grid=grid(49152), stream=stream0)
        buf338 = empty((768, ), device='cuda', dtype=torch.float32)
        buf339 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___14_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf337, squeeze_88, buf338, buf339, 768, 64, grid=grid(768), stream=stream0)
        buf340 = buf330; del buf330  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___14_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_29, buf334, unsqueeze_682, buf338, squeeze_88, buf336, primals_119, buf340, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_29
        del primals_119
        del squeeze_88
        del unsqueeze_682
        buf341 = reinterpret_tensor(buf337, (768, 64), (1, 768), 0); del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf340, buf341, 49152, 128, grid=grid(49152), stream=stream0)
        buf342 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf341, buf342, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf343 = aten.convolution_backward(buf340, add_158, primals_117, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_158
        del primals_117
        buf344 = buf343[0]
        buf345 = buf343[1]
        del buf343
        buf346 = empty((768, ), device='cuda', dtype=torch.float32)
        buf347 = empty((768, ), device='cuda', dtype=torch.float32)
        buf348 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_13_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf334, buf344, convolution_28, unsqueeze_694, squeeze_85, buf346, buf347, buf348, 768, 8192, grid=grid(768), stream=stream0)
        buf349 = buf340; del buf340  # reuse
        # Source Nodes: [l__mod___blocks_13_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_28, buf334, buf344, unsqueeze_694, buf347, squeeze_85, buf346, primals_115, buf349, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf334
        del buf344
        del convolution_28
        del primals_115
        del squeeze_85
        del unsqueeze_694
        buf350 = buf341; del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf349, buf350, 49152, 128, grid=grid(49152), stream=stream0)
        buf351 = buf347; del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf350, buf351, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf352 = aten.convolution_backward(buf349, add_153, primals_113, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_153
        del primals_113
        buf353 = buf352[0]
        buf354 = buf352[1]
        del buf352
        buf355 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf353, buf355, 768, 8192, grid=grid(768), stream=stream0)
        buf356 = reinterpret_tensor(buf350, (768, 64), (64, 1), 0); del buf350  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___13_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf353, convolution_27, unsqueeze_706, buf356, 49152, 128, grid=grid(49152), stream=stream0)
        buf357 = empty((768, ), device='cuda', dtype=torch.float32)
        buf358 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___13_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf356, squeeze_82, buf357, buf358, 768, 64, grid=grid(768), stream=stream0)
        buf359 = buf349; del buf349  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___13_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_27, buf353, unsqueeze_706, buf357, squeeze_82, buf355, primals_111, buf359, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_27
        del primals_111
        del squeeze_82
        del unsqueeze_706
        buf360 = reinterpret_tensor(buf356, (768, 64), (1, 768), 0); del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf359, buf360, 49152, 128, grid=grid(49152), stream=stream0)
        buf361 = buf357; del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf360, buf361, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf362 = aten.convolution_backward(buf359, add_147, primals_109, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_147
        del primals_109
        buf363 = buf362[0]
        buf364 = buf362[1]
        del buf362
        buf365 = empty((768, ), device='cuda', dtype=torch.float32)
        buf366 = empty((768, ), device='cuda', dtype=torch.float32)
        buf367 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_12_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf353, buf363, convolution_26, unsqueeze_718, squeeze_79, buf365, buf366, buf367, 768, 8192, grid=grid(768), stream=stream0)
        buf368 = buf359; del buf359  # reuse
        # Source Nodes: [l__mod___blocks_12_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_26, buf353, buf363, unsqueeze_718, buf366, squeeze_79, buf365, primals_107, buf368, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf353
        del buf363
        del convolution_26
        del primals_107
        del squeeze_79
        del unsqueeze_718
        buf369 = buf360; del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf368, buf369, 49152, 128, grid=grid(49152), stream=stream0)
        buf370 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf369, buf370, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf371 = aten.convolution_backward(buf368, add_142, primals_105, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_142
        del primals_105
        buf372 = buf371[0]
        buf373 = buf371[1]
        del buf371
        buf374 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf372, buf374, 768, 8192, grid=grid(768), stream=stream0)
        buf375 = reinterpret_tensor(buf369, (768, 64), (64, 1), 0); del buf369  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___12_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf372, convolution_25, unsqueeze_730, buf375, 49152, 128, grid=grid(49152), stream=stream0)
        buf376 = empty((768, ), device='cuda', dtype=torch.float32)
        buf377 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___12_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf375, squeeze_76, buf376, buf377, 768, 64, grid=grid(768), stream=stream0)
        buf378 = buf368; del buf368  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___12_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_25, buf372, unsqueeze_730, buf376, squeeze_76, buf374, primals_103, buf378, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_25
        del primals_103
        del squeeze_76
        del unsqueeze_730
        buf379 = reinterpret_tensor(buf375, (768, 64), (1, 768), 0); del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf378, buf379, 49152, 128, grid=grid(49152), stream=stream0)
        buf380 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf379, buf380, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf381 = aten.convolution_backward(buf378, add_136, primals_101, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_136
        del primals_101
        buf382 = buf381[0]
        buf383 = buf381[1]
        del buf381
        buf384 = empty((768, ), device='cuda', dtype=torch.float32)
        buf385 = empty((768, ), device='cuda', dtype=torch.float32)
        buf386 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_11_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf372, buf382, convolution_24, unsqueeze_742, squeeze_73, buf384, buf385, buf386, 768, 8192, grid=grid(768), stream=stream0)
        buf387 = buf378; del buf378  # reuse
        # Source Nodes: [l__mod___blocks_11_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_24, buf372, buf382, unsqueeze_742, buf385, squeeze_73, buf384, primals_99, buf387, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf372
        del buf382
        del convolution_24
        del primals_99
        del squeeze_73
        del unsqueeze_742
        buf388 = buf379; del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf387, buf388, 49152, 128, grid=grid(49152), stream=stream0)
        buf389 = buf385; del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf388, buf389, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf390 = aten.convolution_backward(buf387, add_131, primals_97, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_131
        del primals_97
        buf391 = buf390[0]
        buf392 = buf390[1]
        del buf390
        buf393 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf391, buf393, 768, 8192, grid=grid(768), stream=stream0)
        buf394 = reinterpret_tensor(buf388, (768, 64), (64, 1), 0); del buf388  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___11_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf391, convolution_23, unsqueeze_754, buf394, 49152, 128, grid=grid(49152), stream=stream0)
        buf395 = empty((768, ), device='cuda', dtype=torch.float32)
        buf396 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___11_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf394, squeeze_70, buf395, buf396, 768, 64, grid=grid(768), stream=stream0)
        buf397 = buf387; del buf387  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___11_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_23, buf391, unsqueeze_754, buf395, squeeze_70, buf393, primals_95, buf397, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_23
        del primals_95
        del squeeze_70
        del unsqueeze_754
        buf398 = reinterpret_tensor(buf394, (768, 64), (1, 768), 0); del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf397, buf398, 49152, 128, grid=grid(49152), stream=stream0)
        buf399 = buf395; del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf398, buf399, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf400 = aten.convolution_backward(buf397, add_125, primals_93, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_125
        del primals_93
        buf401 = buf400[0]
        buf402 = buf400[1]
        del buf400
        buf403 = empty((768, ), device='cuda', dtype=torch.float32)
        buf404 = empty((768, ), device='cuda', dtype=torch.float32)
        buf405 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_10_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf391, buf401, convolution_22, unsqueeze_766, squeeze_67, buf403, buf404, buf405, 768, 8192, grid=grid(768), stream=stream0)
        buf406 = buf397; del buf397  # reuse
        # Source Nodes: [l__mod___blocks_10_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_22, buf391, buf401, unsqueeze_766, buf404, squeeze_67, buf403, primals_91, buf406, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf391
        del buf401
        del convolution_22
        del primals_91
        del squeeze_67
        del unsqueeze_766
        buf407 = buf398; del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf406, buf407, 49152, 128, grid=grid(49152), stream=stream0)
        buf408 = buf404; del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf407, buf408, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf409 = aten.convolution_backward(buf406, add_120, primals_89, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_120
        del primals_89
        buf410 = buf409[0]
        buf411 = buf409[1]
        del buf409
        buf412 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf410, buf412, 768, 8192, grid=grid(768), stream=stream0)
        buf413 = reinterpret_tensor(buf407, (768, 64), (64, 1), 0); del buf407  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___10_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf410, convolution_21, unsqueeze_778, buf413, 49152, 128, grid=grid(49152), stream=stream0)
        buf414 = empty((768, ), device='cuda', dtype=torch.float32)
        buf415 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___10_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf413, squeeze_64, buf414, buf415, 768, 64, grid=grid(768), stream=stream0)
        buf416 = buf406; del buf406  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___10_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_21, buf410, unsqueeze_778, buf414, squeeze_64, buf412, primals_87, buf416, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_21
        del primals_87
        del squeeze_64
        del unsqueeze_778
        buf417 = reinterpret_tensor(buf413, (768, 64), (1, 768), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf416, buf417, 49152, 128, grid=grid(49152), stream=stream0)
        buf418 = buf414; del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf417, buf418, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf419 = aten.convolution_backward(buf416, add_114, primals_85, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_114
        del primals_85
        buf420 = buf419[0]
        buf421 = buf419[1]
        del buf419
        buf422 = empty((768, ), device='cuda', dtype=torch.float32)
        buf423 = empty((768, ), device='cuda', dtype=torch.float32)
        buf424 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_9_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf410, buf420, convolution_20, unsqueeze_790, squeeze_61, buf422, buf423, buf424, 768, 8192, grid=grid(768), stream=stream0)
        buf425 = buf416; del buf416  # reuse
        # Source Nodes: [l__mod___blocks_9_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_20, buf410, buf420, unsqueeze_790, buf423, squeeze_61, buf422, primals_83, buf425, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf410
        del buf420
        del convolution_20
        del primals_83
        del squeeze_61
        del unsqueeze_790
        buf426 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf425, buf426, 49152, 128, grid=grid(49152), stream=stream0)
        buf427 = buf423; del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf426, buf427, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf428 = aten.convolution_backward(buf425, add_109, primals_81, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_109
        del primals_81
        buf429 = buf428[0]
        buf430 = buf428[1]
        del buf428
        buf431 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf429, buf431, 768, 8192, grid=grid(768), stream=stream0)
        buf432 = reinterpret_tensor(buf426, (768, 64), (64, 1), 0); del buf426  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___9_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf429, convolution_19, unsqueeze_802, buf432, 49152, 128, grid=grid(49152), stream=stream0)
        buf433 = empty((768, ), device='cuda', dtype=torch.float32)
        buf434 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___9_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf432, squeeze_58, buf433, buf434, 768, 64, grid=grid(768), stream=stream0)
        buf435 = buf425; del buf425  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___9_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_19, buf429, unsqueeze_802, buf433, squeeze_58, buf431, primals_79, buf435, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_19
        del primals_79
        del squeeze_58
        del unsqueeze_802
        buf436 = reinterpret_tensor(buf432, (768, 64), (1, 768), 0); del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf435, buf436, 49152, 128, grid=grid(49152), stream=stream0)
        buf437 = buf433; del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf436, buf437, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf438 = aten.convolution_backward(buf435, add_103, primals_77, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_103
        del primals_77
        buf439 = buf438[0]
        buf440 = buf438[1]
        del buf438
        buf441 = empty((768, ), device='cuda', dtype=torch.float32)
        buf442 = empty((768, ), device='cuda', dtype=torch.float32)
        buf443 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_8_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf429, buf439, convolution_18, unsqueeze_814, squeeze_55, buf441, buf442, buf443, 768, 8192, grid=grid(768), stream=stream0)
        buf444 = buf435; del buf435  # reuse
        # Source Nodes: [l__mod___blocks_8_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_18, buf429, buf439, unsqueeze_814, buf442, squeeze_55, buf441, primals_75, buf444, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf429
        del buf439
        del convolution_18
        del primals_75
        del squeeze_55
        del unsqueeze_814
        buf445 = buf436; del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf444, buf445, 49152, 128, grid=grid(49152), stream=stream0)
        buf446 = buf442; del buf442  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf445, buf446, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf447 = aten.convolution_backward(buf444, add_98, primals_73, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_98
        del primals_73
        buf448 = buf447[0]
        buf449 = buf447[1]
        del buf447
        buf450 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf448, buf450, 768, 8192, grid=grid(768), stream=stream0)
        buf451 = reinterpret_tensor(buf445, (768, 64), (64, 1), 0); del buf445  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf448, convolution_17, unsqueeze_826, buf451, 49152, 128, grid=grid(49152), stream=stream0)
        buf452 = empty((768, ), device='cuda', dtype=torch.float32)
        buf453 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf451, squeeze_52, buf452, buf453, 768, 64, grid=grid(768), stream=stream0)
        buf454 = buf444; del buf444  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_17, buf448, unsqueeze_826, buf452, squeeze_52, buf450, primals_71, buf454, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_17
        del primals_71
        del squeeze_52
        del unsqueeze_826
        buf455 = reinterpret_tensor(buf451, (768, 64), (1, 768), 0); del buf451  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf454, buf455, 49152, 128, grid=grid(49152), stream=stream0)
        buf456 = buf452; del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf455, buf456, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf457 = aten.convolution_backward(buf454, add_92, primals_69, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_92
        del primals_69
        buf458 = buf457[0]
        buf459 = buf457[1]
        del buf457
        buf460 = empty((768, ), device='cuda', dtype=torch.float32)
        buf461 = empty((768, ), device='cuda', dtype=torch.float32)
        buf462 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_7_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf448, buf458, convolution_16, unsqueeze_838, squeeze_49, buf460, buf461, buf462, 768, 8192, grid=grid(768), stream=stream0)
        buf463 = buf454; del buf454  # reuse
        # Source Nodes: [l__mod___blocks_7_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_16, buf448, buf458, unsqueeze_838, buf461, squeeze_49, buf460, primals_67, buf463, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf448
        del buf458
        del convolution_16
        del primals_67
        del squeeze_49
        del unsqueeze_838
        buf464 = buf455; del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf463, buf464, 49152, 128, grid=grid(49152), stream=stream0)
        buf465 = buf461; del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf464, buf465, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf466 = aten.convolution_backward(buf463, add_87, primals_65, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_87
        del primals_65
        buf467 = buf466[0]
        buf468 = buf466[1]
        del buf466
        buf469 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf467, buf469, 768, 8192, grid=grid(768), stream=stream0)
        buf470 = reinterpret_tensor(buf464, (768, 64), (64, 1), 0); del buf464  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf467, convolution_15, unsqueeze_850, buf470, 49152, 128, grid=grid(49152), stream=stream0)
        buf471 = empty((768, ), device='cuda', dtype=torch.float32)
        buf472 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf470, squeeze_46, buf471, buf472, 768, 64, grid=grid(768), stream=stream0)
        buf473 = buf463; del buf463  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_15, buf467, unsqueeze_850, buf471, squeeze_46, buf469, primals_63, buf473, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_15
        del primals_63
        del squeeze_46
        del unsqueeze_850
        buf474 = reinterpret_tensor(buf470, (768, 64), (1, 768), 0); del buf470  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf473, buf474, 49152, 128, grid=grid(49152), stream=stream0)
        buf475 = buf471; del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf474, buf475, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf476 = aten.convolution_backward(buf473, add_81, primals_61, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_81
        del primals_61
        buf477 = buf476[0]
        buf478 = buf476[1]
        del buf476
        buf479 = empty((768, ), device='cuda', dtype=torch.float32)
        buf480 = empty((768, ), device='cuda', dtype=torch.float32)
        buf481 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_6_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf467, buf477, convolution_14, unsqueeze_862, squeeze_43, buf479, buf480, buf481, 768, 8192, grid=grid(768), stream=stream0)
        buf482 = buf473; del buf473  # reuse
        # Source Nodes: [l__mod___blocks_6_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_14, buf467, buf477, unsqueeze_862, buf480, squeeze_43, buf479, primals_59, buf482, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf467
        del buf477
        del convolution_14
        del primals_59
        del squeeze_43
        del unsqueeze_862
        buf483 = buf474; del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf482, buf483, 49152, 128, grid=grid(49152), stream=stream0)
        buf484 = buf480; del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf483, buf484, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf485 = aten.convolution_backward(buf482, add_76, primals_57, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_76
        del primals_57
        buf486 = buf485[0]
        buf487 = buf485[1]
        del buf485
        buf488 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf486, buf488, 768, 8192, grid=grid(768), stream=stream0)
        buf489 = reinterpret_tensor(buf483, (768, 64), (64, 1), 0); del buf483  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf486, convolution_13, unsqueeze_874, buf489, 49152, 128, grid=grid(49152), stream=stream0)
        buf490 = empty((768, ), device='cuda', dtype=torch.float32)
        buf491 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf489, squeeze_40, buf490, buf491, 768, 64, grid=grid(768), stream=stream0)
        buf492 = buf482; del buf482  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_13, buf486, unsqueeze_874, buf490, squeeze_40, buf488, primals_55, buf492, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_13
        del primals_55
        del squeeze_40
        del unsqueeze_874
        buf493 = reinterpret_tensor(buf489, (768, 64), (1, 768), 0); del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf492, buf493, 49152, 128, grid=grid(49152), stream=stream0)
        buf494 = buf490; del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf493, buf494, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf495 = aten.convolution_backward(buf492, add_70, primals_53, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_70
        del primals_53
        buf496 = buf495[0]
        buf497 = buf495[1]
        del buf495
        buf498 = empty((768, ), device='cuda', dtype=torch.float32)
        buf499 = empty((768, ), device='cuda', dtype=torch.float32)
        buf500 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_5_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf486, buf496, convolution_12, unsqueeze_886, squeeze_37, buf498, buf499, buf500, 768, 8192, grid=grid(768), stream=stream0)
        buf501 = buf492; del buf492  # reuse
        # Source Nodes: [l__mod___blocks_5_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_12, buf486, buf496, unsqueeze_886, buf499, squeeze_37, buf498, primals_51, buf501, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf486
        del buf496
        del convolution_12
        del primals_51
        del squeeze_37
        del unsqueeze_886
        buf502 = buf493; del buf493  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf501, buf502, 49152, 128, grid=grid(49152), stream=stream0)
        buf503 = buf499; del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf502, buf503, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf504 = aten.convolution_backward(buf501, add_65, primals_49, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_65
        del primals_49
        buf505 = buf504[0]
        buf506 = buf504[1]
        del buf504
        buf507 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf505, buf507, 768, 8192, grid=grid(768), stream=stream0)
        buf508 = reinterpret_tensor(buf502, (768, 64), (64, 1), 0); del buf502  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf505, convolution_11, unsqueeze_898, buf508, 49152, 128, grid=grid(49152), stream=stream0)
        buf509 = empty((768, ), device='cuda', dtype=torch.float32)
        buf510 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf508, squeeze_34, buf509, buf510, 768, 64, grid=grid(768), stream=stream0)
        buf511 = buf501; del buf501  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_11, buf505, unsqueeze_898, buf509, squeeze_34, buf507, primals_47, buf511, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_11
        del primals_47
        del squeeze_34
        del unsqueeze_898
        buf512 = reinterpret_tensor(buf508, (768, 64), (1, 768), 0); del buf508  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf511, buf512, 49152, 128, grid=grid(49152), stream=stream0)
        buf513 = buf509; del buf509  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf512, buf513, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf514 = aten.convolution_backward(buf511, add_59, primals_45, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_59
        del primals_45
        buf515 = buf514[0]
        buf516 = buf514[1]
        del buf514
        buf517 = empty((768, ), device='cuda', dtype=torch.float32)
        buf518 = empty((768, ), device='cuda', dtype=torch.float32)
        buf519 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_4_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf505, buf515, convolution_10, unsqueeze_910, squeeze_31, buf517, buf518, buf519, 768, 8192, grid=grid(768), stream=stream0)
        buf520 = buf511; del buf511  # reuse
        # Source Nodes: [l__mod___blocks_4_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_10, buf505, buf515, unsqueeze_910, buf518, squeeze_31, buf517, primals_43, buf520, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf505
        del buf515
        del convolution_10
        del primals_43
        del squeeze_31
        del unsqueeze_910
        buf521 = buf512; del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf520, buf521, 49152, 128, grid=grid(49152), stream=stream0)
        buf522 = buf518; del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf521, buf522, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf523 = aten.convolution_backward(buf520, add_54, primals_41, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_54
        del primals_41
        buf524 = buf523[0]
        buf525 = buf523[1]
        del buf523
        buf526 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf524, buf526, 768, 8192, grid=grid(768), stream=stream0)
        buf527 = reinterpret_tensor(buf521, (768, 64), (64, 1), 0); del buf521  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf524, convolution_9, unsqueeze_922, buf527, 49152, 128, grid=grid(49152), stream=stream0)
        buf528 = empty((768, ), device='cuda', dtype=torch.float32)
        buf529 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf527, squeeze_28, buf528, buf529, 768, 64, grid=grid(768), stream=stream0)
        buf530 = buf520; del buf520  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_9, buf524, unsqueeze_922, buf528, squeeze_28, buf526, primals_39, buf530, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_9
        del primals_39
        del squeeze_28
        del unsqueeze_922
        buf531 = reinterpret_tensor(buf527, (768, 64), (1, 768), 0); del buf527  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf530, buf531, 49152, 128, grid=grid(49152), stream=stream0)
        buf532 = buf528; del buf528  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf531, buf532, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf533 = aten.convolution_backward(buf530, add_48, primals_37, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_48
        del primals_37
        buf534 = buf533[0]
        buf535 = buf533[1]
        del buf533
        buf536 = empty((768, ), device='cuda', dtype=torch.float32)
        buf537 = empty((768, ), device='cuda', dtype=torch.float32)
        buf538 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf524, buf534, convolution_8, unsqueeze_934, squeeze_25, buf536, buf537, buf538, 768, 8192, grid=grid(768), stream=stream0)
        buf539 = buf530; del buf530  # reuse
        # Source Nodes: [l__mod___blocks_3_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_8, buf524, buf534, unsqueeze_934, buf537, squeeze_25, buf536, primals_35, buf539, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf524
        del buf534
        del convolution_8
        del primals_35
        del squeeze_25
        del unsqueeze_934
        buf540 = buf531; del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf539, buf540, 49152, 128, grid=grid(49152), stream=stream0)
        buf541 = buf537; del buf537  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf540, buf541, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf542 = aten.convolution_backward(buf539, add_43, primals_33, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_43
        del primals_33
        buf543 = buf542[0]
        buf544 = buf542[1]
        del buf542
        buf545 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf543, buf545, 768, 8192, grid=grid(768), stream=stream0)
        buf546 = reinterpret_tensor(buf540, (768, 64), (64, 1), 0); del buf540  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf543, convolution_7, unsqueeze_946, buf546, 49152, 128, grid=grid(49152), stream=stream0)
        buf547 = empty((768, ), device='cuda', dtype=torch.float32)
        buf548 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf546, squeeze_22, buf547, buf548, 768, 64, grid=grid(768), stream=stream0)
        buf549 = buf539; del buf539  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_7, buf543, unsqueeze_946, buf547, squeeze_22, buf545, primals_31, buf549, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_7
        del primals_31
        del squeeze_22
        del unsqueeze_946
        buf550 = reinterpret_tensor(buf546, (768, 64), (1, 768), 0); del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf549, buf550, 49152, 128, grid=grid(49152), stream=stream0)
        buf551 = buf547; del buf547  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf550, buf551, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf552 = aten.convolution_backward(buf549, add_37, primals_29, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_37
        del primals_29
        buf553 = buf552[0]
        buf554 = buf552[1]
        del buf552
        buf555 = empty((768, ), device='cuda', dtype=torch.float32)
        buf556 = empty((768, ), device='cuda', dtype=torch.float32)
        buf557 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf543, buf553, convolution_6, unsqueeze_958, squeeze_19, buf555, buf556, buf557, 768, 8192, grid=grid(768), stream=stream0)
        buf558 = buf549; del buf549  # reuse
        # Source Nodes: [l__mod___blocks_2_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_6, buf543, buf553, unsqueeze_958, buf556, squeeze_19, buf555, primals_27, buf558, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf543
        del buf553
        del convolution_6
        del primals_27
        del squeeze_19
        del unsqueeze_958
        buf559 = buf550; del buf550  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf558, buf559, 49152, 128, grid=grid(49152), stream=stream0)
        buf560 = buf556; del buf556  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf559, buf560, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf561 = aten.convolution_backward(buf558, add_32, primals_25, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_32
        del primals_25
        buf562 = buf561[0]
        buf563 = buf561[1]
        del buf561
        buf564 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf562, buf564, 768, 8192, grid=grid(768), stream=stream0)
        buf565 = reinterpret_tensor(buf559, (768, 64), (64, 1), 0); del buf559  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf562, convolution_5, unsqueeze_970, buf565, 49152, 128, grid=grid(49152), stream=stream0)
        buf566 = empty((768, ), device='cuda', dtype=torch.float32)
        buf567 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf565, squeeze_16, buf566, buf567, 768, 64, grid=grid(768), stream=stream0)
        buf568 = buf558; del buf558  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_5, buf562, unsqueeze_970, buf566, squeeze_16, buf564, primals_23, buf568, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_5
        del primals_23
        del squeeze_16
        del unsqueeze_970
        buf569 = reinterpret_tensor(buf565, (768, 64), (1, 768), 0); del buf565  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf568, buf569, 49152, 128, grid=grid(49152), stream=stream0)
        buf570 = buf566; del buf566  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf569, buf570, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf571 = aten.convolution_backward(buf568, add_26, primals_21, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_26
        del primals_21
        buf572 = buf571[0]
        buf573 = buf571[1]
        del buf571
        buf574 = empty((768, ), device='cuda', dtype=torch.float32)
        buf575 = empty((768, ), device='cuda', dtype=torch.float32)
        buf576 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf562, buf572, convolution_4, unsqueeze_982, squeeze_13, buf574, buf575, buf576, 768, 8192, grid=grid(768), stream=stream0)
        buf577 = buf568; del buf568  # reuse
        # Source Nodes: [l__mod___blocks_1_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_4, buf562, buf572, unsqueeze_982, buf575, squeeze_13, buf574, primals_19, buf577, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf562
        del buf572
        del convolution_4
        del primals_19
        del squeeze_13
        del unsqueeze_982
        buf578 = buf569; del buf569  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf577, buf578, 49152, 128, grid=grid(49152), stream=stream0)
        buf579 = buf575; del buf575  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf578, buf579, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf580 = aten.convolution_backward(buf577, add_21, primals_17, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_21
        del primals_17
        buf581 = buf580[0]
        buf582 = buf580[1]
        del buf580
        buf583 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf581, buf583, 768, 8192, grid=grid(768), stream=stream0)
        buf584 = reinterpret_tensor(buf578, (768, 64), (64, 1), 0); del buf578  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf581, convolution_3, unsqueeze_994, buf584, 49152, 128, grid=grid(49152), stream=stream0)
        buf585 = empty((768, ), device='cuda', dtype=torch.float32)
        buf586 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf584, squeeze_10, buf585, buf586, 768, 64, grid=grid(768), stream=stream0)
        buf587 = buf577; del buf577  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_3, buf581, unsqueeze_994, buf585, squeeze_10, buf583, primals_15, buf587, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_3
        del primals_15
        del squeeze_10
        del unsqueeze_994
        buf588 = reinterpret_tensor(buf584, (768, 64), (1, 768), 0); del buf584  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf587, buf588, 49152, 128, grid=grid(49152), stream=stream0)
        buf589 = buf585; del buf585  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf588, buf589, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf590 = aten.convolution_backward(buf587, add_15, primals_13, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_15
        del primals_13
        buf591 = buf590[0]
        buf592 = buf590[1]
        del buf590
        buf593 = empty((768, ), device='cuda', dtype=torch.float32)
        buf594 = empty((768, ), device='cuda', dtype=torch.float32)
        buf595 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf581, buf591, convolution_2, unsqueeze_1006, squeeze_7, buf593, buf594, buf595, 768, 8192, grid=grid(768), stream=stream0)
        buf596 = buf587; del buf587  # reuse
        # Source Nodes: [l__mod___blocks_0_2], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution_2, buf581, buf591, unsqueeze_1006, buf594, squeeze_7, buf593, primals_11, buf596, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf581
        del buf591
        del convolution_2
        del primals_11
        del squeeze_7
        del unsqueeze_1006
        buf597 = buf588; del buf588  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf596, buf597, 49152, 128, grid=grid(49152), stream=stream0)
        buf598 = buf594; del buf594  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf597, buf598, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf599 = aten.convolution_backward(buf596, add_10, primals_9, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_10
        del primals_9
        buf600 = buf599[0]
        buf601 = buf599[1]
        del buf599
        buf602 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf600, buf602, 768, 8192, grid=grid(768), stream=stream0)
        buf603 = reinterpret_tensor(buf597, (768, 64), (64, 1), 0); del buf597  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_native_batch_norm_backward_relu_8.run(buf600, convolution_1, unsqueeze_1018, buf603, 49152, 128, grid=grid(49152), stream=stream0)
        buf604 = empty((768, ), device='cuda', dtype=torch.float32)
        buf605 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu]
        triton_per_fused_native_batch_norm_backward_relu_9.run(buf603, squeeze_4, buf604, buf605, 768, 64, grid=grid(768), stream=stream0)
        buf606 = buf596; del buf596  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___fn_1], Original ATen: [aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10.run(convolution_1, buf600, unsqueeze_1018, buf604, squeeze_4, buf602, primals_7, buf606, 8192, 768, grid=grid(8192, 768), stream=stream0)
        del convolution_1
        del primals_7
        del squeeze_4
        del unsqueeze_1018
        buf607 = reinterpret_tensor(buf603, (768, 64), (1, 768), 0); del buf603  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf606, buf607, 49152, 128, grid=grid(49152), stream=stream0)
        buf608 = buf604; del buf604  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf607, buf608, 768, 64, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf609 = aten.convolution_backward(buf606, add_4, primals_5, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_4
        del primals_5
        buf610 = buf609[0]
        buf611 = buf609[1]
        del buf609
        buf612 = empty((768, ), device='cuda', dtype=torch.float32)
        buf613 = empty((768, ), device='cuda', dtype=torch.float32)
        buf614 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___stem_1], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu]
        triton_red_fused_add_native_batch_norm_backward_relu_11.run(buf600, buf610, convolution, unsqueeze_1030, squeeze_1, buf612, buf613, buf614, 768, 8192, grid=grid(768), stream=stream0)
        buf615 = buf606; del buf606  # reuse
        # Source Nodes: [l__mod___stem_1], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12.run(convolution, buf600, buf610, unsqueeze_1030, buf613, squeeze_1, buf612, primals_3, buf615, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf600
        del buf610
        del convolution
        del primals_3
        del squeeze_1
        del unsqueeze_1030
        buf616 = buf607; del buf607  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_5.run(buf615, buf616, 49152, 128, grid=grid(49152), stream=stream0)
        buf617 = buf613; del buf613  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_6.run(buf616, buf617, 768, 64, grid=grid(768), stream=stream0)
        del buf616
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf618 = aten.convolution_backward(buf615, primals_458, primals_1, [768], [7, 7], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf615
        del primals_1
        del primals_458
        buf619 = buf618[1]
        return (buf619, buf617, buf614, buf612, buf611, buf608, buf605, buf602, buf601, buf598, buf595, buf593, buf592, buf589, buf586, buf583, buf582, buf579, buf576, buf574, buf573, buf570, buf567, buf564, buf563, buf560, buf557, buf555, buf554, buf551, buf548, buf545, buf544, buf541, buf538, buf536, buf535, buf532, buf529, buf526, buf525, buf522, buf519, buf517, buf516, buf513, buf510, buf507, buf506, buf503, buf500, buf498, buf497, buf494, buf491, buf488, buf487, buf484, buf481, buf479, buf478, buf475, buf472, buf469, buf468, buf465, buf462, buf460, buf459, buf456, buf453, buf450, buf449, buf446, buf443, buf441, buf440, buf437, buf434, buf431, buf430, buf427, buf424, buf422, buf421, buf418, buf415, buf412, buf411, buf408, buf405, buf403, buf402, buf399, buf396, buf393, buf392, buf389, buf386, buf384, buf383, buf380, buf377, buf374, buf373, buf370, buf367, buf365, buf364, buf361, buf358, buf355, buf354, buf351, buf348, buf346, buf345, buf342, buf339, buf336, buf335, buf332, buf329, buf327, buf326, buf323, buf320, buf317, buf316, buf313, buf310, buf308, buf307, buf304, buf301, buf298, buf297, buf294, buf291, buf289, buf288, buf285, buf282, buf279, buf278, buf275, buf272, buf270, buf269, buf266, buf263, buf260, buf259, buf256, buf253, buf251, buf250, buf247, buf244, buf241, buf240, buf237, buf234, buf232, buf231, buf228, buf225, buf222, buf221, buf218, buf215, buf213, buf212, buf209, buf206, buf203, buf202, buf199, buf196, buf194, buf193, buf190, buf187, buf184, buf183, buf180, buf177, buf175, buf174, buf171, buf168, buf165, buf164, buf161, buf158, buf156, buf155, buf152, buf149, buf146, buf145, buf142, buf139, buf137, buf136, buf133, buf130, buf127, buf126, buf123, buf120, buf118, buf117, buf114, buf111, buf108, buf107, buf104, buf101, buf99, buf98, buf95, buf92, buf89, buf88, buf85, buf82, buf80, buf79, buf76, buf73, buf70, buf69, buf66, buf63, buf61, buf60, buf57, buf54, buf51, buf50, buf47, buf44, buf42, buf41, buf38, buf35, buf32, buf31, buf28, buf25, buf23, buf22, buf19, buf16, buf13, buf12, buf9, buf6, buf3, reinterpret_tensor(buf1, (1000, 768), (768, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, 3, 7, 7), (147, 1, 21, 3), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_4 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_10 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_15 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_21 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_26 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_32 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_37 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_43 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_48 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_54 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_59 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_65 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_70 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_76 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_81 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_87 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_92 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_98 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_103 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_109 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_114 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_120 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_125 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_131 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_136 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_142 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_147 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_153 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_158 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_164 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_169 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_175 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_180 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_186 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_191 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_197 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_202 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_208 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_213 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_219 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_224 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_230 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_235 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_241 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_246 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_252 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_257 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_263 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_268 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_274 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_279 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_285 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_157 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_290 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_160 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_296 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_163 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_301 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_166 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_307 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_169 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_312 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_57 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_172 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_318 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_58 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_175 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_323 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_59 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_178 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_329 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_181 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_334 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_184 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_340 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_62 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_187 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_345 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_63 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_190 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_351 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    convolution_64 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda:0', dtype=torch.float32)
    squeeze_193 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((8, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_274 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_286 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_358 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_370 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_382 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_406 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_418 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_430 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_454 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_466 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_478 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_502 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_514 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_526 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_538 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_550 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_562 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_574 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_586 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_598 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_610 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_622 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_634 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_646 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_658 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_670 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_682 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_694 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_706 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_718 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_730 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_742 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_754 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_766 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_778 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_790 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_802 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_814 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_826 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_838 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_850 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_862 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_874 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_886 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_898 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_910 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_922 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_934 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_946 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_958 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_970 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_982 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_994 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1006 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1018 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1030 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_191, primals_193, primals_195, primals_197, primals_199, primals_201, primals_203, primals_205, primals_207, primals_209, primals_211, primals_213, primals_215, primals_217, primals_219, primals_221, primals_223, primals_225, primals_227, primals_229, primals_231, primals_233, primals_235, primals_237, primals_239, primals_241, primals_243, primals_245, primals_247, primals_249, primals_251, primals_253, primals_255, primals_257, primals_259, primals_458, convolution, squeeze_1, add_4, convolution_1, squeeze_4, add_10, convolution_2, squeeze_7, add_15, convolution_3, squeeze_10, add_21, convolution_4, squeeze_13, add_26, convolution_5, squeeze_16, add_32, convolution_6, squeeze_19, add_37, convolution_7, squeeze_22, add_43, convolution_8, squeeze_25, add_48, convolution_9, squeeze_28, add_54, convolution_10, squeeze_31, add_59, convolution_11, squeeze_34, add_65, convolution_12, squeeze_37, add_70, convolution_13, squeeze_40, add_76, convolution_14, squeeze_43, add_81, convolution_15, squeeze_46, add_87, convolution_16, squeeze_49, add_92, convolution_17, squeeze_52, add_98, convolution_18, squeeze_55, add_103, convolution_19, squeeze_58, add_109, convolution_20, squeeze_61, add_114, convolution_21, squeeze_64, add_120, convolution_22, squeeze_67, add_125, convolution_23, squeeze_70, add_131, convolution_24, squeeze_73, add_136, convolution_25, squeeze_76, add_142, convolution_26, squeeze_79, add_147, convolution_27, squeeze_82, add_153, convolution_28, squeeze_85, add_158, convolution_29, squeeze_88, add_164, convolution_30, squeeze_91, add_169, convolution_31, squeeze_94, add_175, convolution_32, squeeze_97, add_180, convolution_33, squeeze_100, add_186, convolution_34, squeeze_103, add_191, convolution_35, squeeze_106, add_197, convolution_36, squeeze_109, add_202, convolution_37, squeeze_112, add_208, convolution_38, squeeze_115, add_213, convolution_39, squeeze_118, add_219, convolution_40, squeeze_121, add_224, convolution_41, squeeze_124, add_230, convolution_42, squeeze_127, add_235, convolution_43, squeeze_130, add_241, convolution_44, squeeze_133, add_246, convolution_45, squeeze_136, add_252, convolution_46, squeeze_139, add_257, convolution_47, squeeze_142, add_263, convolution_48, squeeze_145, add_268, convolution_49, squeeze_148, add_274, convolution_50, squeeze_151, add_279, convolution_51, squeeze_154, add_285, convolution_52, squeeze_157, add_290, convolution_53, squeeze_160, add_296, convolution_54, squeeze_163, add_301, convolution_55, squeeze_166, add_307, convolution_56, squeeze_169, add_312, convolution_57, squeeze_172, add_318, convolution_58, squeeze_175, add_323, convolution_59, squeeze_178, add_329, convolution_60, squeeze_181, add_334, convolution_61, squeeze_184, add_340, convolution_62, squeeze_187, add_345, convolution_63, squeeze_190, add_351, convolution_64, squeeze_193, clone, permute_1, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, unsqueeze_850, unsqueeze_862, unsqueeze_874, unsqueeze_886, unsqueeze_898, unsqueeze_910, unsqueeze_922, unsqueeze_934, unsqueeze_946, unsqueeze_958, unsqueeze_970, unsqueeze_982, unsqueeze_994, unsqueeze_1006, unsqueeze_1018, unsqueeze_1030, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('convmixer_768_32', benchmark_compiled_module)
