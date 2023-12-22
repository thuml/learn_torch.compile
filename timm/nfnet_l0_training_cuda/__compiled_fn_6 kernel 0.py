
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


# kernel path: /tmp/torchinductor_youkaichao/55/c55a772m5zf7yzop57cpvysia4jhpezy6oivfmwubqkqjalyqp6l.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_div_fill_mul_sigmoid_sub_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_sigmoid_sub_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp1 = 49.0
    tmp2 = tmp0 / tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = 1.0
    tmp6 = tmp5 - tmp4
    tmp7 = tmp3 * tmp6
    tmp8 = tmp7 + tmp5
    tmp9 = tmp4 * tmp8
    tmp10 = tmp2 * tmp9
    tl.store(out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hv/chvrpot6r73or5rngegay4nxryua54cqibt2bffbhh7522gdvuup.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 2304
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (112896*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3a/c3apbtazj3kk6znji732gumm75ig7bzaqcm3qrfvlp6q5y6m7d5x.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_red_fused_mul_native_batch_norm_backward_view_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_view_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2304
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tmp12 - tmp5
        tmp14 = 0.0006510416666666666
        tmp15 = tmp9 * tmp14
        tmp17 = tmp16 * tmp16
        tmp18 = tmp15 * tmp17
        tmp19 = tmp13 * tmp18
        tmp20 = tmp11 - tmp19
        tmp21 = tmp2 * tmp14
        tmp22 = tmp20 - tmp21
        tmp24 = 0.04562504637317021
        tmp25 = tmp23 * tmp24
        tmp26 = tmp16 * tmp25
        tmp27 = tmp22 * tmp26
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp27, rmask & xmask)
    tmp28 = tmp9 * tmp16
    tmp29 = 0.04562504637317021
    tmp30 = tmp28 * tmp29
    tl.store(out_ptr3 + (x0), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ex/cexc7bil7dgyrqvahcxqxscikfk3yo5pw3mcs4i3nuu4m4rurjio.py
# Source Nodes: [sigmoid_11], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_11 => sigmoid_62
triton_per_fused_mul_sigmoid_sigmoid_backward_sum_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_sum_4', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
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
    tmp5 = tl.load(in_ptr1 + (r1 + (49*x0)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.2
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = 1.0
    tmp14 = tmp13 - tmp12
    tmp15 = tmp12 * tmp14
    tmp16 = tmp10 * tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wx/cwxr23phie4id5vv4lo7n5tf5jg73bitbr4mhyc2tdl5lwmythzg.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 8
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/35/c35pfacv6xautckfjflu6c64wsbnelillokpiifkrjnpnbpan3ob.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a6/ca6amfrbnqqeyrppksazpunqwiqrkwzh64cvty3b52c546xy6qrl.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_7', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eg/cegphme5yvnxfnkln32vscyl2md7gmsq2ipyb2tmaf4hypp2vmqu.py
# Source Nodes: [sigmoid_11], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
# sigmoid_11 => sigmoid_62
triton_poi_fused_add_div_mul_sigmoid_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sigmoid_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.2
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp4 * tmp6
    tmp9 = 49.0
    tmp10 = tmp8 / tmp9
    tmp11 = tmp7 + tmp10
    tl.store(out_ptr0 + (x2), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vb/cvbci4w3n6i6vicsp2zuvpvxmfkalbmagw6gwmese5zlzc3vhbim.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 1536
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (75264*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ex/cexbrsubm3mpmn3ofq5u5a6p367uweo3wd6xkdwfrwmfgd6mxcq4.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1536
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
    tmp5 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 0.0026041666666666665
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.09125009274634042
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ir/cirvoowniox37bb3wpupeld4sgywkzg3rsjluzsxyojzpisoxanb.py
# Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act3], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
# getattr_getattr_l__mod___stages___3_____2___act3 => sigmoid_61
triton_poi_fused_add_fill_mul_silu_sub_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_silu_sub_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o6/co62qq7xjkzzb2c3vevlbvf547hx6g632kdelhqqi7xe3clxlyd2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 384
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (18816*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w6/cw66kgebu2oltsqhp2lt4duaj6pkj6myqsu5br67kreb6x5oasc3.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 384
    XBLOCK: tl.constexpr = 1
    rnumel = 576
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (576*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (576*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 0.001736111111111111
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.07450538873672485
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (576*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lx/clx7p36thtudbsfx6vga2xsckbsi3maeuvoj4wmewrf2l7utcrxd.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_red_fused_mul_native_batch_norm_backward_view_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_view_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tmp12 - tmp5
        tmp14 = 0.0006510416666666666
        tmp15 = tmp9 * tmp14
        tmp17 = tmp16 * tmp16
        tmp18 = tmp15 * tmp17
        tmp19 = tmp13 * tmp18
        tmp20 = tmp11 - tmp19
        tmp21 = tmp2 * tmp14
        tmp22 = tmp20 - tmp21
        tmp24 = 0.04562504637317021
        tmp25 = tmp23 * tmp24
        tmp26 = tmp16 * tmp25
        tmp27 = tmp22 * tmp26
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp27, rmask & xmask)
    tmp28 = tmp9 * tmp16
    tmp29 = 0.04562504637317021
    tmp30 = tmp28 * tmp29
    tl.store(out_ptr3 + (x0), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sv/csvhqlo3hx7uh2ouxi2mfmsejovsloh2m2fnzvebcxhillfswbki.py
# Source Nodes: [sigmoid_10], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_10 => sigmoid_57
triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
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
    tmp4 = tl.load(in_ptr2 + (r1 + (49*x0)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (49*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = 0.9622504486493761
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = 0.2
    tmp8 = tmp6 * tmp7
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = 1.0
    tmp20 = tmp19 - tmp18
    tmp21 = tmp18 * tmp20
    tmp22 = tmp16 * tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/74/c74ak3fzau5c2qflehu3cxoga4r6kaugeclcc5ngf3kpr6y67evx.py
# Source Nodes: [sigmoid_10], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
# sigmoid_10 => sigmoid_57
triton_poi_fused_add_div_mul_sigmoid_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sigmoid_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x2), None)
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.9622504486493761
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = 0.2
    tmp8 = tmp6 * tmp7
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp10 * tmp12
    tmp15 = 49.0
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2b/c2bja3ze2vaztoj6fykbtlt4ivteyzlwml7arcdmlv37amffyxys.py
# Source Nodes: [sigmoid_9], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_9 => sigmoid_52
triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_17', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
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
    tmp4 = tl.load(in_ptr2 + (r1 + (49*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_out_ptr0 + (r1 + (49*x0)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (r1 + (49*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr4 + (r1 + (49*x0)), rmask, other=0.0)
    tmp23 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = 0.9622504486493761
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = 0.9805806756909201
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp15 = 2.0
    tmp16 = tmp14 * tmp15
    tmp18 = tmp16 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp24 = tl.sigmoid(tmp23)
    tmp25 = 1.0
    tmp26 = tmp25 - tmp24
    tmp27 = tmp24 * tmp26
    tmp28 = tmp22 * tmp27
    tl.store(in_out_ptr0 + (r1 + (49*x0)), tmp12, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fv/cfvtwdefq642icvcrfjd2bvnldky5tbitaigttwchjp5ffsrjoo7.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xg/cxgsvwmnydflgheqssiuxyunf52ndks5hxsjwwqxeljtxyfy3a3c.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lk/clk3ddthqcfm7bafcqdtabrrnf5wdsr7a65yumsd7yqkeqwzm4dg.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_red_fused_mul_native_batch_norm_backward_view_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_view_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tmp12 - tmp5
        tmp14 = 0.0006510416666666666
        tmp15 = tmp9 * tmp14
        tmp17 = tmp16 * tmp16
        tmp18 = tmp15 * tmp17
        tmp19 = tmp13 * tmp18
        tmp20 = tmp11 - tmp19
        tmp21 = tmp2 * tmp14
        tmp22 = tmp20 - tmp21
        tmp24 = 0.04562504637317021
        tmp25 = tmp23 * tmp24
        tmp26 = tmp16 * tmp25
        tmp27 = tmp22 * tmp26
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp27, rmask & xmask)
    tmp28 = tmp9 * tmp16
    tmp29 = 0.04562504637317021
    tmp30 = tmp28 * tmp29
    tl.store(out_ptr3 + (x0), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7x/c7xm3yzofi5kfu576wjdnlmw6d6patfunypwykhrffzecwgxwwxj.py
# Source Nodes: [sigmoid_8], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_8 => sigmoid_47
triton_red_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_sum_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_sum_21', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 14
        r2 = (rindex // 14)
        tmp0 = tl.load(in_ptr0 + (r3 + (196*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((7*(tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(7, 1 + (r2 // 2)))))) + (7*(tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(7, 1 + (r2 // 2))))) >= 0, 0, 7))) + (49*x0) + (tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(7, 1 + (r1 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(7, 1 + (r1 // 2))))) >= 0, 0, 7))), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr2 + (r3 + (196*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r3 + (196*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp1 / 4
        tmp3 = tl.math.max(0, (r2 // 2))
        tmp4 = tl.math.min(7, 1 + (r2 // 2))
        tmp5 = tmp3 < tmp4
        tmp6 = tl.math.max(0, (r1 // 2))
        tmp7 = tl.math.min(7, 1 + (r1 // 2))
        tmp8 = tmp6 < tmp7
        tmp9 = tmp5 & tmp8
        tmp10 = 0.0
        tmp11 = tl.where(tmp9, tmp2, tmp10)
        tmp12 = tmp0 + tmp11
        tmp13 = 0.8980265101338745
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 * tmp15
        tmp17 = 0.2
        tmp18 = tmp16 * tmp17
        tmp19 = 2.0
        tmp20 = tmp18 * tmp19
        tmp22 = tmp20 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask, tmp25, _tmp24)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp26 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.sigmoid(tmp26)
    tmp28 = 1.0
    tmp29 = tmp28 - tmp27
    tmp30 = tmp27 * tmp29
    tmp31 = tmp24 * tmp30
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yz/cyzu4fk3iz7awr67rusir7t5hpzike564npd2htdn4vxchqzlqkl.py
# Source Nodes: [sigmoid_8], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.div, aten.mul, aten.sigmoid]
# sigmoid_8 => sigmoid_47
triton_poi_fused_add_avg_pool2d_backward_div_mul_sigmoid_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_div_mul_sigmoid_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + ((7*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(7, 1 + (x1 // 2)))))) + (7*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(7, 1 + (x1 // 2))))) >= 0, 0, 7))) + (49*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) >= 0, 0, 7))), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (x3), None)
    tmp21 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp1 / 4
    tmp3 = tl.math.max(0, (x1 // 2))
    tmp4 = tl.math.min(7, 1 + (x1 // 2))
    tmp5 = tmp3 < tmp4
    tmp6 = tl.math.max(0, (x0 // 2))
    tmp7 = tl.math.min(7, 1 + (x0 // 2))
    tmp8 = tmp6 < tmp7
    tmp9 = tmp5 & tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp2, tmp10)
    tmp12 = tmp0 + tmp11
    tmp13 = 0.8980265101338745
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 * tmp15
    tmp17 = 0.2
    tmp18 = tmp16 * tmp17
    tmp19 = 2.0
    tmp20 = tmp18 * tmp19
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = tmp20 * tmp22
    tmp25 = 196.0
    tmp26 = tmp24 / tmp25
    tmp27 = tmp23 + tmp26
    tl.store(out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dr/cdrkybs7qa3gy3ajm5os44ngminngsz565hajoaek4ix2oraprrm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (301056*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/to/ctooaltzznv3bzsg7tqqrkbtqj6tcsjcvlkbrvu4fl2c6d2hyvxi.py
# Source Nodes: [sigmoid_7], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_7 => sigmoid_42
triton_per_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_sum_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_sum_24', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex
    r1 = rindex % 14
    r2 = (rindex // 14)
    tmp0 = tl.load(in_out_ptr0 + (r3 + (196*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + ((7*(tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(7, 1 + (r2 // 2)))))) + (7*(tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(7, 1 + (r2 // 2))))) >= 0, 0, 7))) + (49*x0) + (tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(7, 1 + (r1 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(7, 1 + (r1 // 2))))) >= 0, 0, 7))), rmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr1 + (r3 + (196*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr2 + (r3 + (196*x0)), rmask, other=0.0)
    tmp20 = tl.load(in_ptr3 + (r3 + (196*x0)), rmask, other=0.0)
    tmp27 = tl.load(in_ptr4 + (r3 + (196*x0)), rmask, other=0.0)
    tmp33 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1 / 4
    tmp3 = tl.math.max(0, (r2 // 2))
    tmp4 = tl.math.min(7, 1 + (r2 // 2))
    tmp5 = tmp3 < tmp4
    tmp6 = tl.math.max(0, (r1 // 2))
    tmp7 = tl.math.min(7, 1 + (r1 // 2))
    tmp8 = tmp6 < tmp7
    tmp9 = tmp5 & tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp2, tmp10)
    tmp12 = tmp0 + tmp11
    tmp13 = 0.8980265101338745
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 * tmp15
    tmp18 = 0.9128709291752768
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 * tmp20
    tmp22 = tmp16 + tmp21
    tmp23 = 0.2
    tmp24 = tmp22 * tmp23
    tmp25 = 2.0
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(rmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp34 = tl.sigmoid(tmp33)
    tmp35 = 1.0
    tmp36 = tmp35 - tmp34
    tmp37 = tmp34 * tmp36
    tmp38 = tmp32 * tmp37
    tl.store(in_out_ptr0 + (r3 + (196*x0)), tmp22, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp38, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ei/ceiuwa2hayxvhbydoun7vml5diho2n4vuu2wkormmueuciq6tx4k.py
# Source Nodes: [sigmoid_7], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
# sigmoid_7 => sigmoid_42
triton_poi_fused_add_div_mul_sigmoid_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sigmoid_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.2
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp4 * tmp6
    tmp9 = 196.0
    tmp10 = tmp8 / tmp9
    tmp11 = tmp7 + tmp10
    tl.store(out_ptr0 + (x2), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ve/cvewdiw5vhstrlg5qnsakdkuh5qpezc32476kqhgafwxyozvcco7.py
# Source Nodes: [sigmoid_6], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_6 => sigmoid_37
triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_26', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (196*x0)), rmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (196*x0)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (196*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = 0.9284766908852592
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = 0.2
    tmp8 = tmp6 * tmp7
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = 1.0
    tmp20 = tmp19 - tmp18
    tmp21 = tmp18 * tmp20
    tmp22 = tmp16 * tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pp/cppg3sd7ajerobow3eyx45fpfriqo6tb4pvzxquulhrdutzkalj4.py
# Source Nodes: [sigmoid_6], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
# sigmoid_6 => sigmoid_37
triton_poi_fused_add_div_mul_sigmoid_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sigmoid_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x2), None)
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.9284766908852592
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = 0.2
    tmp8 = tmp6 * tmp7
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp10 * tmp12
    tmp15 = 196.0
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/et/cetz56aglxyvh4tihzfqgll4zvzptttkptv2mea47ddv5lvbapi5.py
# Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_5 => sigmoid_32
triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_28', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + (196*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (r1 + (196*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (196*x0)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (r1 + (196*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr4 + (r1 + (196*x0)), rmask, other=0.0)
    tmp23 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = 0.9284766908852592
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = 0.9449111825230679
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp15 = 2.0
    tmp16 = tmp14 * tmp15
    tmp18 = tmp16 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp24 = tl.sigmoid(tmp23)
    tmp25 = 1.0
    tmp26 = tmp25 - tmp24
    tmp27 = tmp24 * tmp26
    tmp28 = tmp22 * tmp27
    tl.store(in_out_ptr0 + (r1 + (196*x0)), tmp12, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mp/cmpcmtnujy2s5p6aj6qznkeovhw6ye5qfghngztajsixjqgmwdh7.py
# Source Nodes: [sigmoid_4], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_4 => sigmoid_27
triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_29', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (196*x0)), rmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (196*x0)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (196*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = 0.9622504486493761
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = 0.2
    tmp8 = tmp6 * tmp7
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = 1.0
    tmp20 = tmp19 - tmp18
    tmp21 = tmp18 * tmp20
    tmp22 = tmp16 * tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/c5/cc5bnjjyo3oviy26yfqybtv6cqh7nu357vb7mqtzsjomqoeigwa2.py
# Source Nodes: [sigmoid_4], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
# sigmoid_4 => sigmoid_27
triton_poi_fused_add_div_mul_sigmoid_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sigmoid_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x2), None)
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.9622504486493761
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = 0.2
    tmp8 = tmp6 * tmp7
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp10 * tmp12
    tmp15 = 196.0
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5p/c5psf2xdfuqyipj6ltobjxoad7p23p3cmbejtdwlcyrbfwtdtbgu.py
# Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_3 => sigmoid_22
triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_31', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + (196*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (r1 + (196*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (196*x0)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (r1 + (196*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr4 + (r1 + (196*x0)), rmask, other=0.0)
    tmp23 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = 0.9622504486493761
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = 0.9805806756909201
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp15 = 2.0
    tmp16 = tmp14 * tmp15
    tmp18 = tmp16 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp24 = tl.sigmoid(tmp23)
    tmp25 = 1.0
    tmp26 = tmp25 - tmp24
    tmp27 = tmp24 * tmp26
    tmp28 = tmp22 * tmp27
    tl.store(in_out_ptr0 + (r1 + (196*x0)), tmp12, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmn4h6ppnfivnwkgotekqfjsrmzca5wuhsxud626yp7drfjofsx.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_32', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5x/c5xhcoxg5uucdaur4icbpoy455gu3y6xlnz34uto25sc7tlac56m.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (301056*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cc/cccdn6zkn4bqhxlbhtesundjqbmoxfaibi6kwxa57pftj4oeb6zv.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 384
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
    tmp5 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 0.001953125
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.07902489841601695
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c2/cc23guunbn7yazomh5goejos26n5zoo5oxs4b7gd5palmisalmsf.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1536
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
    tmp5 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 0.001953125
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.07902489841601695
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hp/chpayb7t2mjotlk6ns5csaf7hwc4ldgphncfcdrgwvfsfhg76rbb.py
# Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_2 => sigmoid_17
triton_per_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_sum_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_sum_36', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):
    xnumel = 4096
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
    r1 = rindex % 28
    r2 = (rindex // 28)
    tmp0 = tl.load(in_ptr0 + (r3 + (784*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + ((14*(tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(14, 1 + (r2 // 2)))))) + (14*(tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(14, 1 + (r2 // 2))))) >= 0, 0, 14))) + (196*x0) + (tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(14, 1 + (r1 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(14, 1 + (r1 // 2))))) >= 0, 0, 14))), rmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr2 + (r3 + (784*x0)), rmask, other=0.0)
    tmp21 = tl.load(in_ptr3 + (r3 + (784*x0)), rmask, other=0.0)
    tmp27 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1 / 4
    tmp3 = tl.math.max(0, (r2 // 2))
    tmp4 = tl.math.min(14, 1 + (r2 // 2))
    tmp5 = tmp3 < tmp4
    tmp6 = tl.math.max(0, (r1 // 2))
    tmp7 = tl.math.min(14, 1 + (r1 // 2))
    tmp8 = tmp6 < tmp7
    tmp9 = tmp5 & tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp2, tmp10)
    tmp12 = tmp0 + tmp11
    tmp13 = 0.9622504486493761
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 * tmp15
    tmp17 = 0.2
    tmp18 = tmp16 * tmp17
    tmp19 = 2.0
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp28 = tl.sigmoid(tmp27)
    tmp29 = 1.0
    tmp30 = tmp29 - tmp28
    tmp31 = tmp28 * tmp30
    tmp32 = tmp26 * tmp31
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp32, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/o6/co6dkdo3gczt4yjnhnwml6otlnkklwbhx5opa6chownk4mtipofy.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/5f/c5f6mt6xc7sgu4gxekugmj3v2t24olot3quxreybnuxsp35pla33.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ot/cotv7c4m3fh2xh27l43jixvgknud32jg3i7hexwcdur3fn2fev7i.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/nq/cnqrk72wmcxmpct7pbzofgho5as27jdxaxohfkahbjutyembskdx.py
# Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.div, aten.mul, aten.sigmoid]
# sigmoid_2 => sigmoid_17
triton_poi_fused_add_avg_pool2d_backward_div_mul_sigmoid_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_div_mul_sigmoid_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 28
    x1 = (xindex // 28) % 28
    x2 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + ((14*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2)))))) + (14*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2))))) >= 0, 0, 14))) + (196*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) >= 0, 0, 14))), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (x3), None)
    tmp21 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp1 / 4
    tmp3 = tl.math.max(0, (x1 // 2))
    tmp4 = tl.math.min(14, 1 + (x1 // 2))
    tmp5 = tmp3 < tmp4
    tmp6 = tl.math.max(0, (x0 // 2))
    tmp7 = tl.math.min(14, 1 + (x0 // 2))
    tmp8 = tmp6 < tmp7
    tmp9 = tmp5 & tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp2, tmp10)
    tmp12 = tmp0 + tmp11
    tmp13 = 0.9622504486493761
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 * tmp15
    tmp17 = 0.2
    tmp18 = tmp16 * tmp17
    tmp19 = 2.0
    tmp20 = tmp18 * tmp19
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = tmp20 * tmp22
    tmp25 = 784.0
    tmp26 = tmp24 / tmp25
    tmp27 = tmp23 + tmp26
    tl.store(out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u4/cu4ozb7mxxxgi4dkkwrtkdeii6gszz4wh6l3hfqei7f6vp35kuyu.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_41', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/zj/czjma5wsu2rvdes7iv6woqmfyua5doywij3szeyhqboyvsdonenz.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
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
    tmp5 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = 0.0078125
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.1580497968320339
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/io/ciongmqbpwq6evgmtepxcpfld2bprxd6iplrka2cbvjylnxmbawv.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act3], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
# getattr_getattr_l__mod___stages___1_____1___act3 => sigmoid_16
triton_poi_fused_add_fill_mul_silu_sub_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_silu_sub_43', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s7/cs7f4sxosbbcuesj3gsjj5prh3potnb7b42bnc7ev3267fcscqlf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u5/cu5rwmx47s3pikvvmf7pdjqaeioafbvdk4vytabznglefjhchxfl.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 576
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (576*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (576*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 0.001736111111111111
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.07450538873672485
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (576*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cf/ccf5425va3xpwkpizi3gxht2g45dwqbkvu5tp2kbza6yxbu6oz6f.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 128
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
    tmp5 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 0.001953125
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.07902489841601695
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ao/caob4yawpq3zrjdocn65vxe7laqqwwnivla464jazmpwdrvjsmdw.py
# Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_1 => sigmoid_12
triton_per_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_sum_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_sum_47', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel):
    xnumel = 4096
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
    r1 = rindex % 28
    r2 = (rindex // 28)
    tmp0 = tl.load(in_out_ptr0 + (r3 + (784*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + ((14*(tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(14, 1 + (r2 // 2)))))) + (14*(tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(14, 1 + (r2 // 2))))) >= 0, 0, 14))) + (196*x0) + (tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(14, 1 + (r1 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(14, 1 + (r1 // 2))))) >= 0, 0, 14))), rmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr1 + (r3 + (784*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr2 + (r3 + (784*x0)), rmask, other=0.0)
    tmp20 = tl.load(in_ptr3 + (r3 + (784*x0)), rmask, other=0.0)
    tmp27 = tl.load(in_ptr4 + (r3 + (784*x0)), rmask, other=0.0)
    tmp33 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1 / 4
    tmp3 = tl.math.max(0, (r2 // 2))
    tmp4 = tl.math.min(14, 1 + (r2 // 2))
    tmp5 = tmp3 < tmp4
    tmp6 = tl.math.max(0, (r1 // 2))
    tmp7 = tl.math.min(14, 1 + (r1 // 2))
    tmp8 = tmp6 < tmp7
    tmp9 = tmp5 & tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp2, tmp10)
    tmp12 = tmp0 + tmp11
    tmp13 = 0.9622504486493761
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 * tmp15
    tmp18 = 0.9805806756909201
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 * tmp20
    tmp22 = tmp16 + tmp21
    tmp23 = 0.2
    tmp24 = tmp22 * tmp23
    tmp25 = 2.0
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp34 = tl.sigmoid(tmp33)
    tmp35 = 1.0
    tmp36 = tmp35 - tmp34
    tmp37 = tmp34 * tmp36
    tmp38 = tmp32 * tmp37
    tl.store(in_out_ptr0 + (r3 + (784*x0)), tmp22, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp38, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gx/cgxyabxvxgipbdrhmxrobpgfjhot77y5vifcss6bsan76f3hzijy.py
# Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
# sigmoid_1 => sigmoid_12
triton_poi_fused_add_div_mul_sigmoid_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sigmoid_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.2
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp4 * tmp6
    tmp9 = 784.0
    tmp10 = tmp8 / tmp9
    tmp11 = tmp7 + tmp10
    tl.store(out_ptr0 + (x2), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6q/c6qcxwpv5evpu3y3phwq446kfpate7spsk5rymk4zb7xq42pumj5.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_49', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ux/cuxoi5ms33vi6rbyp5jsfkcce2k2ccr4mgmjqusks5wzysm6gmjt.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
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
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j7/cj72fyaj4awqlvybs74ylqchwrxmps66zzi7inscsgoxq25hsddf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_51', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ty/ctyx72lqo6klwehho5obx56zsx5z2gs5hv27fqefkegviuqjsuru.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp5 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 0.00390625
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.11175808310508728
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kv/ckv2jt4jlake2x7jzpqg5mtsjfmxkmnfxgod2g6mp3n5xmp42vga.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 512
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
    tmp5 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 0.00390625
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.11175808310508728
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t5/ct5mbb4f5bnwakiuyqd6dkm7nzopylgu6uy5e7lsmlycaq4d6jrn.py
# Source Nodes: [sigmoid], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid => sigmoid_7
triton_red_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_sum_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_sum_54', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 56
        r2 = (rindex // 56)
        tmp0 = tl.load(in_out_ptr0 + (r3 + (3136*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + ((28*(tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(28, 1 + (r2 // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(28, 1 + (r2 // 2))))) >= 0, 0, 28))) + (784*x0) + (tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(28, 1 + (r1 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(28, 1 + (r1 // 2))))) >= 0, 0, 28))), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr1 + (r3 + (3136*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr2 + (r3 + (3136*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp1 / 4
        tmp3 = tl.math.max(0, (r2 // 2))
        tmp4 = tl.math.min(28, 1 + (r2 // 2))
        tmp5 = tmp3 < tmp4
        tmp6 = tl.math.max(0, (r1 // 2))
        tmp7 = tl.math.min(28, 1 + (r1 // 2))
        tmp8 = tmp6 < tmp7
        tmp9 = tmp5 & tmp8
        tmp10 = 0.0
        tmp11 = tl.where(tmp9, tmp2, tmp10)
        tmp12 = tmp0 + tmp11
        tmp13 = 0.9805806756909201
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 * tmp15
        tmp17 = 0.2
        tmp18 = tmp16 * tmp17
        tmp19 = 2.0
        tmp20 = tmp18 * tmp19
        tmp22 = tmp20 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask, tmp25, _tmp24)
        tl.store(in_out_ptr0 + (r3 + (3136*x0)), tmp16, rmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp26 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.sigmoid(tmp26)
    tmp28 = 1.0
    tmp29 = tmp28 - tmp27
    tmp30 = tmp27 * tmp29
    tmp31 = tmp24 * tmp30
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vy/cvyxt6krth4vmtaybi3uwulpjfvxiamt5tpz4spkdcco5wzdas6u.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/wt/cwtncmxpzvpfjqgvuts3rr4laaezvyzlyjq7uosggdfb2lubgyk5.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_56', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/la/clazu3yzj4wdq4a6lqzwr2csub3yceqrda5d6olxngsfpwj6w77e.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 8
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/d2/cd27vmhvtb2o5uvvvrlih57ekjzcpxmxthfwswpy3qg3lyujdejy.py
# Source Nodes: [sigmoid], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
# sigmoid => sigmoid_7
triton_poi_fused_add_div_mul_sigmoid_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sigmoid_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.2
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp4 * tmp6
    tmp9 = 3136.0
    tmp10 = tmp8 / tmp9
    tmp11 = tmp7 + tmp10
    tl.store(out_ptr0 + (x2), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qd/cqd3f4e46hv572brqoas5zuz7srwxwm5rnvgowlxtwwoyqddgakc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_59', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/3v/c3vj4y45senx2pj6w26rosjnst24sx3w2wq3crkz3mcftte4d3mv.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr1 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = 0.015625
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.22351616621017456
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2r/c2rl5ac5fb5nu4beaodgepkywav3zn45q7lbnn5zaz33qr4rw4sf.py
# Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act3], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
# getattr_getattr_l__mod___stages___0_____0___act3 => sigmoid_6
triton_poi_fused_add_fill_mul_silu_sub_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_silu_sub_61', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/y2/cy2x25l3x3ewib7evxggfg5csdx6an6xdzoa3xl42drrllvib25e.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qq/cqqf5ycpsxquyb3x6xfxnj3kcuwotg7ixcvn2dy3mgmp27cfop5n.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_63', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/bf/cbfff6tysgok7lyutt3yetxjzkf4eblsaby62oxyspiat6a7pz44.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 576
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (576*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (576*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 0.001736111111111111
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.07450538873672485
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (576*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ix/cixilr77zvrmnydl6zm2imajlorg7odr4wbccmmcka45izl5bxt3.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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
    tmp5 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = 0.0078125
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.1580497968320339
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7t/c7tvrwsqhmxsayj3sgelb32efm2r6jylcvnotkthuuh4ypznsleb.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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
    tmp5 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = 0.0078125
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.1580497968320339
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6e/c6ewu33an762qoa2awrpczkolaaewgoeotj745crwbg6hygjdvz4.py
# Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act1], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
# getattr_getattr_l__mod___stages___0_____0___act1 => sigmoid_3
triton_poi_fused_add_fill_mul_silu_sub_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_silu_sub_67', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp5 = tl.load(in_ptr1 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp3 - tmp6
    tmp8 = tmp5 * tmp7
    tmp9 = tmp8 + tmp3
    tmp10 = tmp6 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x0), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/il/cili3ur7u5kd7g7knjguho43hmtg6sjs6cnbbbgwbe33fowydlbv.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_68', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hn/chn3nsnl5mehyuwtwreek6o3zgvuxgdl3dicx3rgrku4aesfmd3x.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
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
        tmp3 = tl.load(in_ptr0 + ((12544*x1) + (802816*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pr/cpr5h7vs3bbedhq7kt5wx4zna2t7deqpyespa6nkekbuxjsyrhv2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3f/c3fupl2ua2wmpjz26y7tptsowni3xbqf3m6r5q4c3gbdaqgeq3cg.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 288
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (288*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (288*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 0.003472222222222222
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.10536653122135592
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (288*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/47/c47cdkishn5ugchzlzsv6maijdwq3vcezpbxz75gjj7r6kuulse6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 416
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
        tmp3 = tl.load(in_ptr0 + ((12544*x1) + (401408*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oz/coznpfmgjuql42dwodewx55uui7qp4f3ypz5azpyrsleym7w7l5b.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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


# kernel path: /tmp/torchinductor_youkaichao/dx/cdxczgqbcevoinn4uwyqubfk7jsorktgnhtsvoz2pwup2s4jsvoa.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_74 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (144*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = 0.006944444444444444
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.1490107774734497
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (144*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ym/cymwbvc7jkm6ih5iqouwupnauh3wue7fdqfqlctbosg5r6ufzc5c.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_75', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/a2/ca2pg545hm27hyesecalwjrtcasxwhsahsb3v4isxxaf57crmwox.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_76', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/dd/cddy7odn7psmyyam7m4a6ozef2dmqbruu3mhd42opb24qhf3rlul.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 27
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (27*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (27*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = 0.037037037037037035
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.34412564994580647
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (27*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_174, primals_176, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, primals_204, primals_206, primals_208, primals_210, primals_212, primals_214, primals_216, primals_218, primals_222, squeeze_1, view_2, convolution, mul_3, squeeze_3, view_5, convolution_1, mul_7, squeeze_5, view_8, convolution_2, mul_11, squeeze_7, view_11, convolution_3, mul_16, squeeze_9, view_14, squeeze_11, view_17, convolution_5, mul_23, squeeze_13, view_20, convolution_6, mul_27, squeeze_15, view_23, convolution_7, mul_31, squeeze_17, view_26, convolution_8, mean, relu, convolution_10, mul_39, avg_pool2d, squeeze_19, view_29, squeeze_21, view_32, convolution_12, mul_46, squeeze_23, view_35, convolution_13, mul_50, squeeze_25, view_38, convolution_14, mul_54, squeeze_27, view_41, convolution_15, mean_1, relu_1, convolution_17, mul_62, squeeze_29, view_44, convolution_18, mul_66, squeeze_31, view_47, convolution_19, mul_70, squeeze_33, view_50, convolution_20, mul_74, squeeze_35, view_53, convolution_21, mean_2, relu_2, convolution_23, mul_82, avg_pool2d_1, squeeze_37, view_56, squeeze_39, view_59, convolution_25, mul_89, squeeze_41, view_62, convolution_26, mul_93, squeeze_43, view_65, convolution_27, mul_97, squeeze_45, view_68, convolution_28, mean_3, relu_3, convolution_30, mul_105, squeeze_47, view_71, convolution_31, mul_109, squeeze_49, view_74, convolution_32, mul_113, squeeze_51, view_77, convolution_33, mul_117, squeeze_53, view_80, convolution_34, mean_4, relu_4, convolution_36, mul_125, squeeze_55, view_83, convolution_37, mul_129, squeeze_57, view_86, convolution_38, mul_133, squeeze_59, view_89, convolution_39, mul_137, squeeze_61, view_92, convolution_40, mean_5, relu_5, convolution_42, mul_145, squeeze_63, view_95, convolution_43, mul_149, squeeze_65, view_98, convolution_44, mul_153, squeeze_67, view_101, convolution_45, mul_157, squeeze_69, view_104, convolution_46, mean_6, relu_6, convolution_48, mul_165, squeeze_71, view_107, convolution_49, mul_169, squeeze_73, view_110, convolution_50, mul_173, squeeze_75, view_113, convolution_51, mul_177, squeeze_77, view_116, convolution_52, mean_7, relu_7, convolution_54, mul_185, squeeze_79, view_119, convolution_55, mul_189, squeeze_81, view_122, convolution_56, mul_193, squeeze_83, view_125, convolution_57, mul_197, squeeze_85, view_128, convolution_58, mean_8, relu_8, convolution_60, mul_205, avg_pool2d_2, squeeze_87, view_131, squeeze_89, view_134, convolution_62, mul_212, squeeze_91, view_137, convolution_63, mul_216, squeeze_93, view_140, convolution_64, mul_220, squeeze_95, view_143, convolution_65, mean_9, relu_9, convolution_67, mul_228, squeeze_97, view_146, convolution_68, mul_232, squeeze_99, view_149, convolution_69, mul_236, squeeze_101, view_152, convolution_70, mul_240, squeeze_103, view_155, convolution_71, mean_10, relu_10, convolution_73, mul_248, squeeze_105, view_158, convolution_74, mul_252, squeeze_107, view_161, convolution_75, mul_256, squeeze_109, view_164, convolution_76, mul_260, squeeze_111, view_167, convolution_77, mean_11, relu_11, convolution_79, add_67, squeeze_113, view_170, convolution_80, clone_28, permute_1, unsqueeze_58, unsqueeze_66, unsqueeze_74, unsqueeze_82, unsqueeze_90, mul_341, unsqueeze_98, unsqueeze_106, unsqueeze_114, unsqueeze_122, mul_400, unsqueeze_130, unsqueeze_138, unsqueeze_146, unsqueeze_154, unsqueeze_162, mul_469, unsqueeze_170, unsqueeze_178, unsqueeze_186, unsqueeze_194, mul_528, unsqueeze_202, unsqueeze_210, unsqueeze_218, unsqueeze_226, mul_587, unsqueeze_234, unsqueeze_242, unsqueeze_250, unsqueeze_258, mul_646, unsqueeze_266, unsqueeze_274, unsqueeze_282, unsqueeze_290, mul_705, unsqueeze_298, unsqueeze_306, unsqueeze_314, unsqueeze_322, mul_764, unsqueeze_330, unsqueeze_338, unsqueeze_346, unsqueeze_354, unsqueeze_362, mul_833, unsqueeze_370, unsqueeze_378, unsqueeze_386, unsqueeze_394, mul_892, unsqueeze_402, unsqueeze_410, unsqueeze_418, unsqueeze_426, unsqueeze_434, mul_961, unsqueeze_442, unsqueeze_450, unsqueeze_458, unsqueeze_466, unsqueeze_474, unsqueeze_482, unsqueeze_490, unsqueeze_498, unsqueeze_506, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (16, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_4, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_5, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_7, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_8, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_10, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_11, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_13, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_14, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_16, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_17, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_19, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_20, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_22, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_23, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_25, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_26, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_28, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_29, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_31, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_32, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_34, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_35, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_37, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_38, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_40, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_41, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_43, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_44, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_46, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_47, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_49, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_50, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_52, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_53, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_55, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_56, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_58, (384, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_59, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_61, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_62, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_64, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_65, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_67, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_68, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_70, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_71, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_73, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_74, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_76, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_77, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_79, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_80, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_82, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_83, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_85, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_86, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_88, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_89, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_91, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_92, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_94, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_95, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_97, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_98, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_100, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_101, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_103, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_104, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_106, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_107, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_109, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_110, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_112, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_113, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_115, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_116, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_118, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_119, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_121, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_122, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_124, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_125, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_127, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_128, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_130, (1536, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_131, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_133, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_134, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_136, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_137, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_139, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_140, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_142, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_143, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_145, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_146, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_148, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_149, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_151, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_152, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_154, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_155, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_157, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_158, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_160, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_161, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_163, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_164, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_166, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_167, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_169, (2304, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_170, (2304, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_172, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_174, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_176, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_178, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_180, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_182, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_184, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_186, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_188, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_190, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_192, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_194, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_196, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_198, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_200, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_202, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_204, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_206, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_208, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_210, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_212, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_214, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_216, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_218, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_222, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(squeeze_1, (16, ), (1, ))
    assert_size_stride(view_2, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(convolution, (8, 16, 112, 112), (200704, 12544, 112, 1))
    assert_size_stride(mul_3, (8, 16, 112, 112), (200704, 12544, 112, 1))
    assert_size_stride(squeeze_3, (32, ), (1, ))
    assert_size_stride(view_5, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(convolution_1, (8, 32, 112, 112), (401408, 12544, 112, 1))
    assert_size_stride(mul_7, (8, 32, 112, 112), (401408, 12544, 112, 1))
    assert_size_stride(squeeze_5, (64, ), (1, ))
    assert_size_stride(view_8, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(convolution_2, (8, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(mul_11, (8, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(squeeze_7, (128, ), (1, ))
    assert_size_stride(view_11, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_3, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(mul_16, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(squeeze_9, (256, ), (1, ))
    assert_size_stride(view_14, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(squeeze_11, (64, ), (1, ))
    assert_size_stride(view_17, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(convolution_5, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(mul_23, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(squeeze_13, (64, ), (1, ))
    assert_size_stride(view_20, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_6, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(mul_27, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(squeeze_15, (64, ), (1, ))
    assert_size_stride(view_23, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_7, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(mul_31, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(squeeze_17, (256, ), (1, ))
    assert_size_stride(view_26, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(convolution_8, (8, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(mean, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(relu, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(convolution_10, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_39, (8, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(avg_pool2d, (8, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(squeeze_19, (512, ), (1, ))
    assert_size_stride(view_29, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(squeeze_21, (128, ), (1, ))
    assert_size_stride(view_32, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_12, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(mul_46, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(squeeze_23, (128, ), (1, ))
    assert_size_stride(view_35, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_13, (8, 128, 28, 28), (100352, 784, 28, 1))
    assert_size_stride(mul_50, (8, 128, 28, 28), (100352, 784, 28, 1))
    assert_size_stride(squeeze_25, (128, ), (1, ))
    assert_size_stride(view_38, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_14, (8, 128, 28, 28), (100352, 784, 28, 1))
    assert_size_stride(mul_54, (8, 128, 28, 28), (100352, 784, 28, 1))
    assert_size_stride(squeeze_27, (512, ), (1, ))
    assert_size_stride(view_41, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(convolution_15, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(mean_1, (8, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(relu_1, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(convolution_17, (8, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_62, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(squeeze_29, (128, ), (1, ))
    assert_size_stride(view_44, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(convolution_18, (8, 128, 28, 28), (100352, 784, 28, 1))
    assert_size_stride(mul_66, (8, 128, 28, 28), (100352, 784, 28, 1))
    assert_size_stride(squeeze_31, (128, ), (1, ))
    assert_size_stride(view_47, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_19, (8, 128, 28, 28), (100352, 784, 28, 1))
    assert_size_stride(mul_70, (8, 128, 28, 28), (100352, 784, 28, 1))
    assert_size_stride(squeeze_33, (128, ), (1, ))
    assert_size_stride(view_50, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_20, (8, 128, 28, 28), (100352, 784, 28, 1))
    assert_size_stride(mul_74, (8, 128, 28, 28), (100352, 784, 28, 1))
    assert_size_stride(squeeze_35, (512, ), (1, ))
    assert_size_stride(view_53, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(convolution_21, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(mean_2, (8, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(relu_2, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(convolution_23, (8, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_82, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(avg_pool2d_1, (8, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(squeeze_37, (1536, ), (1, ))
    assert_size_stride(view_56, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(squeeze_39, (384, ), (1, ))
    assert_size_stride(view_59, (384, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(convolution_25, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(mul_89, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(squeeze_41, (384, ), (1, ))
    assert_size_stride(view_62, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_26, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(mul_93, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_43, (384, ), (1, ))
    assert_size_stride(view_65, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_27, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(mul_97, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_45, (1536, ), (1, ))
    assert_size_stride(view_68, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(convolution_28, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(mean_3, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(relu_3, (8, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(convolution_30, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_105, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(squeeze_47, (384, ), (1, ))
    assert_size_stride(view_71, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(convolution_31, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(mul_109, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_49, (384, ), (1, ))
    assert_size_stride(view_74, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_32, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(mul_113, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_51, (384, ), (1, ))
    assert_size_stride(view_77, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_33, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(mul_117, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_53, (1536, ), (1, ))
    assert_size_stride(view_80, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(convolution_34, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(mean_4, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(relu_4, (8, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(convolution_36, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_125, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(squeeze_55, (384, ), (1, ))
    assert_size_stride(view_83, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(convolution_37, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(mul_129, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_57, (384, ), (1, ))
    assert_size_stride(view_86, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_38, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(mul_133, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_59, (384, ), (1, ))
    assert_size_stride(view_89, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_39, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(mul_137, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_61, (1536, ), (1, ))
    assert_size_stride(view_92, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(convolution_40, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(mean_5, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(relu_5, (8, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(convolution_42, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_145, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(squeeze_63, (384, ), (1, ))
    assert_size_stride(view_95, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(convolution_43, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(mul_149, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_65, (384, ), (1, ))
    assert_size_stride(view_98, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_44, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(mul_153, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_67, (384, ), (1, ))
    assert_size_stride(view_101, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_45, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(mul_157, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_69, (1536, ), (1, ))
    assert_size_stride(view_104, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(convolution_46, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(mean_6, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(relu_6, (8, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(convolution_48, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_165, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(squeeze_71, (384, ), (1, ))
    assert_size_stride(view_107, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(convolution_49, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(mul_169, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_73, (384, ), (1, ))
    assert_size_stride(view_110, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_50, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(mul_173, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_75, (384, ), (1, ))
    assert_size_stride(view_113, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_51, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(mul_177, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_77, (1536, ), (1, ))
    assert_size_stride(view_116, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(convolution_52, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(mean_7, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(relu_7, (8, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(convolution_54, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_185, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(squeeze_79, (384, ), (1, ))
    assert_size_stride(view_119, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(convolution_55, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(mul_189, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_81, (384, ), (1, ))
    assert_size_stride(view_122, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_56, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(mul_193, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_83, (384, ), (1, ))
    assert_size_stride(view_125, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_57, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(mul_197, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_85, (1536, ), (1, ))
    assert_size_stride(view_128, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(convolution_58, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(mean_8, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(relu_8, (8, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(convolution_60, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_205, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(avg_pool2d_2, (8, 1536, 7, 7), (75264, 49, 7, 1))
    assert_size_stride(squeeze_87, (1536, ), (1, ))
    assert_size_stride(view_131, (1536, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(squeeze_89, (384, ), (1, ))
    assert_size_stride(view_134, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(convolution_62, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(mul_212, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_91, (384, ), (1, ))
    assert_size_stride(view_137, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_63, (8, 384, 7, 7), (18816, 49, 7, 1))
    assert_size_stride(mul_216, (8, 384, 7, 7), (18816, 49, 7, 1))
    assert_size_stride(squeeze_93, (384, ), (1, ))
    assert_size_stride(view_140, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_64, (8, 384, 7, 7), (18816, 49, 7, 1))
    assert_size_stride(mul_220, (8, 384, 7, 7), (18816, 49, 7, 1))
    assert_size_stride(squeeze_95, (1536, ), (1, ))
    assert_size_stride(view_143, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(convolution_65, (8, 1536, 7, 7), (75264, 49, 7, 1))
    assert_size_stride(mean_9, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(relu_9, (8, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(convolution_67, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_228, (8, 1536, 7, 7), (75264, 49, 7, 1))
    assert_size_stride(squeeze_97, (384, ), (1, ))
    assert_size_stride(view_146, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(convolution_68, (8, 384, 7, 7), (18816, 49, 7, 1))
    assert_size_stride(mul_232, (8, 384, 7, 7), (18816, 49, 7, 1))
    assert_size_stride(squeeze_99, (384, ), (1, ))
    assert_size_stride(view_149, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_69, (8, 384, 7, 7), (18816, 49, 7, 1))
    assert_size_stride(mul_236, (8, 384, 7, 7), (18816, 49, 7, 1))
    assert_size_stride(squeeze_101, (384, ), (1, ))
    assert_size_stride(view_152, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_70, (8, 384, 7, 7), (18816, 49, 7, 1))
    assert_size_stride(mul_240, (8, 384, 7, 7), (18816, 49, 7, 1))
    assert_size_stride(squeeze_103, (1536, ), (1, ))
    assert_size_stride(view_155, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(convolution_71, (8, 1536, 7, 7), (75264, 49, 7, 1))
    assert_size_stride(mean_10, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(relu_10, (8, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(convolution_73, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_248, (8, 1536, 7, 7), (75264, 49, 7, 1))
    assert_size_stride(squeeze_105, (384, ), (1, ))
    assert_size_stride(view_158, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(convolution_74, (8, 384, 7, 7), (18816, 49, 7, 1))
    assert_size_stride(mul_252, (8, 384, 7, 7), (18816, 49, 7, 1))
    assert_size_stride(squeeze_107, (384, ), (1, ))
    assert_size_stride(view_161, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_75, (8, 384, 7, 7), (18816, 49, 7, 1))
    assert_size_stride(mul_256, (8, 384, 7, 7), (18816, 49, 7, 1))
    assert_size_stride(squeeze_109, (384, ), (1, ))
    assert_size_stride(view_164, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_76, (8, 384, 7, 7), (18816, 49, 7, 1))
    assert_size_stride(mul_260, (8, 384, 7, 7), (18816, 49, 7, 1))
    assert_size_stride(squeeze_111, (1536, ), (1, ))
    assert_size_stride(view_167, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(convolution_77, (8, 1536, 7, 7), (75264, 49, 7, 1))
    assert_size_stride(mean_11, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(relu_11, (8, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(convolution_79, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(add_67, (8, 1536, 7, 7), (75264, 49, 7, 1))
    assert_size_stride(squeeze_113, (2304, ), (1, ))
    assert_size_stride(view_170, (2304, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(convolution_80, (8, 2304, 7, 7), (112896, 49, 7, 1))
    assert_size_stride(clone_28, (8, 2304), (2304, 1))
    assert_size_stride(permute_1, (1000, 2304), (2304, 1))
    assert_size_stride(unsqueeze_58, (1, 2304, 1), (2304, 1, 1))
    assert_size_stride(unsqueeze_66, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_74, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_82, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_90, (1, 384, 1), (384, 1, 1))
    assert_size_stride(mul_341, (8, 1536, 7, 7), (75264, 49, 7, 1))
    assert_size_stride(unsqueeze_98, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_106, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_114, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_122, (1, 384, 1), (384, 1, 1))
    assert_size_stride(mul_400, (8, 1536, 7, 7), (75264, 49, 7, 1))
    assert_size_stride(unsqueeze_130, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_138, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_146, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_154, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_162, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(mul_469, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(unsqueeze_170, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_178, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_186, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_194, (1, 384, 1), (384, 1, 1))
    assert_size_stride(mul_528, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(unsqueeze_202, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_210, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_218, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_226, (1, 384, 1), (384, 1, 1))
    assert_size_stride(mul_587, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(unsqueeze_234, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_242, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_250, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 384, 1), (384, 1, 1))
    assert_size_stride(mul_646, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(unsqueeze_266, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_274, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_282, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_290, (1, 384, 1), (384, 1, 1))
    assert_size_stride(mul_705, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(unsqueeze_298, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_306, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_314, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_322, (1, 384, 1), (384, 1, 1))
    assert_size_stride(mul_764, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(unsqueeze_330, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_338, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_346, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_354, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_362, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(mul_833, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(unsqueeze_370, (1, 512, 1), (512, 1, 1))
    assert_size_stride(unsqueeze_378, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_386, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_394, (1, 128, 1), (128, 1, 1))
    assert_size_stride(mul_892, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(unsqueeze_402, (1, 512, 1), (512, 1, 1))
    assert_size_stride(unsqueeze_410, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_418, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_426, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_434, (1, 512, 1), (512, 1, 1))
    assert_size_stride(mul_961, (8, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(unsqueeze_442, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_450, (1, 64, 1), (64, 1, 1))
    assert_size_stride(unsqueeze_458, (1, 64, 1), (64, 1, 1))
    assert_size_stride(unsqueeze_466, (1, 64, 1), (64, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_482, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_490, (1, 64, 1), (64, 1, 1))
    assert_size_stride(unsqueeze_498, (1, 32, 1), (32, 1, 1))
    assert_size_stride(unsqueeze_506, (1, 16, 1), (16, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_28, out=buf1)
        del clone_28
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty((8, 2304, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_sigmoid_sub_1.run(buf0, convolution_80, buf3, 903168, grid=grid(903168), stream=stream0)
        del convolution_80
        buf4 = empty((2304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_2.run(buf3, buf4, 2304, 392, grid=grid(2304), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf5 = aten.convolution_backward(buf3, add_67, view_170, [2304], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_67
        del buf3
        del view_170
        buf6 = buf5[0]
        buf7 = buf5[1]
        del buf5
        buf11 = empty((2304, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf10 = empty((2304, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_3.run(buf7, primals_169, unsqueeze_58, squeeze_113, primals_170, buf11, buf10, 2304, 1536, grid=grid(2304), stream=stream0)
        del buf7
        del primals_169
        del primals_170
        del squeeze_113
        del unsqueeze_58
        buf12 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf13 = reinterpret_tensor(buf12, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf12  # reuse
        # Source Nodes: [sigmoid_11], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_4.run(buf13, buf6, convolution_77, convolution_79, 12288, 49, grid=grid(12288), stream=stream0)
        del convolution_77
        buf14 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_5.run(buf13, buf14, 1536, 8, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf15 = aten.convolution_backward(buf13, relu_11, primals_218, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf13
        del primals_218
        buf16 = buf15[0]
        buf17 = buf15[1]
        del buf15
        buf18 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_6.run(buf18, relu_11, 3072, grid=grid(3072), stream=stream0)
        del relu_11
        buf19 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_7.run(buf18, buf19, 384, 8, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf20 = aten.convolution_backward(buf18, mean_11, primals_216, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf18
        del mean_11
        del primals_216
        buf21 = buf20[0]
        buf22 = buf20[1]
        del buf20
        buf23 = empty((8, 1536, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_11], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_mul_sigmoid_8.run(buf6, convolution_79, buf21, buf23, 602112, grid=grid(602112), stream=stream0)
        del convolution_79
        buf24 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_9.run(buf23, buf24, 1536, 392, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf25 = aten.convolution_backward(buf23, mul_260, view_167, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_260
        del view_167
        buf26 = buf25[0]
        buf27 = buf25[1]
        del buf25
        buf31 = empty((1536, 384, 1, 1), device='cuda', dtype=torch.float32)
        buf30 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_10.run(buf27, primals_166, unsqueeze_66, squeeze_111, primals_167, buf31, buf30, 1536, 384, grid=grid(1536), stream=stream0)
        del primals_166
        del primals_167
        del squeeze_111
        del unsqueeze_66
        buf32 = buf26; del buf26  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act3], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_11.run(buf32, convolution_76, 150528, grid=grid(150528), stream=stream0)
        del convolution_76
        buf33 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf32, buf33, 384, 392, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf34 = aten.convolution_backward(buf32, mul_256, view_164, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del buf32
        del mul_256
        del view_164
        buf35 = buf34[0]
        buf36 = buf34[1]
        del buf34
        buf40 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf39 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_13.run(buf36, primals_163, unsqueeze_74, squeeze_109, primals_164, buf40, buf39, 384, 576, grid=grid(384), stream=stream0)
        del primals_163
        del primals_164
        del squeeze_109
        del unsqueeze_74
        buf41 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_11.run(buf41, convolution_75, 150528, grid=grid(150528), stream=stream0)
        del convolution_75
        buf42 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf41, buf42, 384, 392, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf43 = aten.convolution_backward(buf41, mul_252, view_161, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del buf41
        del mul_252
        del view_161
        buf44 = buf43[0]
        buf45 = buf43[1]
        del buf43
        buf49 = buf36; del buf36  # reuse
        buf48 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_13.run(buf45, primals_160, unsqueeze_82, squeeze_107, primals_161, buf49, buf48, 384, 576, grid=grid(384), stream=stream0)
        del primals_160
        del primals_161
        del squeeze_107
        del unsqueeze_82
        buf50 = buf44; del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_11.run(buf50, convolution_74, 150528, grid=grid(150528), stream=stream0)
        del convolution_74
        buf51 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf50, buf51, 384, 392, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf52 = aten.convolution_backward(buf50, mul_248, view_158, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf50
        del mul_248
        del view_158
        buf53 = buf52[0]
        buf54 = buf52[1]
        del buf52
        buf58 = reinterpret_tensor(buf27, (384, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf27  # reuse
        buf57 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_14.run(buf54, primals_157, unsqueeze_90, squeeze_105, primals_158, buf58, buf57, 384, 1536, grid=grid(384), stream=stream0)
        del primals_157
        del primals_158
        del squeeze_105
        del unsqueeze_90
        buf59 = reinterpret_tensor(buf21, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf21  # reuse
        buf60 = reinterpret_tensor(buf59, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf59  # reuse
        # Source Nodes: [sigmoid_10], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_15.run(buf60, buf6, buf53, mul_341, convolution_71, convolution_73, 12288, 49, grid=grid(12288), stream=stream0)
        del convolution_71
        buf61 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_5.run(buf60, buf61, 1536, 8, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf62 = aten.convolution_backward(buf60, relu_10, primals_214, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf60
        del primals_214
        buf63 = buf62[0]
        buf64 = buf62[1]
        del buf62
        buf65 = buf63; del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_6.run(buf65, relu_10, 3072, grid=grid(3072), stream=stream0)
        del relu_10
        buf66 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_7.run(buf65, buf66, 384, 8, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf67 = aten.convolution_backward(buf65, mean_10, primals_212, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf65
        del mean_10
        del primals_212
        buf68 = buf67[0]
        buf69 = buf67[1]
        del buf67
        buf70 = buf23; del buf23  # reuse
        # Source Nodes: [sigmoid_10], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_mul_sigmoid_16.run(buf6, buf53, mul_341, convolution_73, buf68, buf70, 602112, grid=grid(602112), stream=stream0)
        del convolution_73
        buf71 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_9.run(buf70, buf71, 1536, 392, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf72 = aten.convolution_backward(buf70, mul_240, view_155, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf70
        del mul_240
        del view_155
        buf73 = buf72[0]
        buf74 = buf72[1]
        del buf72
        buf78 = reinterpret_tensor(buf54, (1536, 384, 1, 1), (384, 1, 1, 1), 0); del buf54  # reuse
        buf77 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_10.run(buf74, primals_154, unsqueeze_98, squeeze_103, primals_155, buf78, buf77, 1536, 384, grid=grid(1536), stream=stream0)
        del primals_154
        del primals_155
        del squeeze_103
        del unsqueeze_98
        buf79 = buf73; del buf73  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act3], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_11.run(buf79, convolution_70, 150528, grid=grid(150528), stream=stream0)
        del convolution_70
        buf80 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf79, buf80, 384, 392, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf81 = aten.convolution_backward(buf79, mul_236, view_152, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del buf79
        del mul_236
        del view_152
        buf82 = buf81[0]
        buf83 = buf81[1]
        del buf81
        buf87 = buf45; del buf45  # reuse
        buf86 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_13.run(buf83, primals_151, unsqueeze_106, squeeze_101, primals_152, buf87, buf86, 384, 576, grid=grid(384), stream=stream0)
        del primals_151
        del primals_152
        del squeeze_101
        del unsqueeze_106
        buf88 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_11.run(buf88, convolution_69, 150528, grid=grid(150528), stream=stream0)
        del convolution_69
        buf89 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf88, buf89, 384, 392, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf90 = aten.convolution_backward(buf88, mul_232, view_149, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del buf88
        del mul_232
        del view_149
        buf91 = buf90[0]
        buf92 = buf90[1]
        del buf90
        buf96 = buf83; del buf83  # reuse
        buf95 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_13.run(buf92, primals_148, unsqueeze_114, squeeze_99, primals_149, buf96, buf95, 384, 576, grid=grid(384), stream=stream0)
        del primals_148
        del primals_149
        del squeeze_99
        del unsqueeze_114
        buf97 = buf91; del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_11.run(buf97, convolution_68, 150528, grid=grid(150528), stream=stream0)
        del convolution_68
        buf98 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf97, buf98, 384, 392, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf99 = aten.convolution_backward(buf97, mul_228, view_146, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf97
        del mul_228
        del view_146
        buf100 = buf99[0]
        buf101 = buf99[1]
        del buf99
        buf105 = reinterpret_tensor(buf74, (384, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf74  # reuse
        buf104 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_14.run(buf101, primals_145, unsqueeze_122, squeeze_97, primals_146, buf105, buf104, 384, 1536, grid=grid(384), stream=stream0)
        del primals_145
        del primals_146
        del squeeze_97
        del unsqueeze_122
        buf106 = buf100; del buf100  # reuse
        buf107 = reinterpret_tensor(buf68, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf68  # reuse
        buf108 = reinterpret_tensor(buf107, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf107  # reuse
        # Source Nodes: [sigmoid_9], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_17.run(buf106, buf108, buf6, buf53, mul_341, mul_400, convolution_65, convolution_67, 12288, 49, grid=grid(12288), stream=stream0)
        del buf53
        del convolution_65
        del mul_341
        del mul_400
        buf109 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_5.run(buf108, buf109, 1536, 8, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf110 = aten.convolution_backward(buf108, relu_9, primals_210, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf108
        del primals_210
        buf111 = buf110[0]
        buf112 = buf110[1]
        del buf110
        buf113 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_6.run(buf113, relu_9, 3072, grid=grid(3072), stream=stream0)
        del relu_9
        buf114 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_7.run(buf113, buf114, 384, 8, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf115 = aten.convolution_backward(buf113, mean_9, primals_208, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf113
        del mean_9
        del primals_208
        buf116 = buf115[0]
        buf117 = buf115[1]
        del buf115
        buf118 = buf6; del buf6  # reuse
        # Source Nodes: [sigmoid_9], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_mul_sigmoid_8.run(buf106, convolution_67, buf116, buf118, 602112, grid=grid(602112), stream=stream0)
        del convolution_67
        buf119 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_9.run(buf118, buf119, 1536, 392, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf120 = aten.convolution_backward(buf118, mul_220, view_143, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf118
        del mul_220
        del view_143
        buf121 = buf120[0]
        buf122 = buf120[1]
        del buf120
        buf126 = reinterpret_tensor(buf101, (1536, 384, 1, 1), (384, 1, 1, 1), 0); del buf101  # reuse
        buf125 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_10.run(buf122, primals_142, unsqueeze_130, squeeze_95, primals_143, buf126, buf125, 1536, 384, grid=grid(1536), stream=stream0)
        del primals_142
        del primals_143
        del squeeze_95
        del unsqueeze_130
        buf127 = buf121; del buf121  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act3], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_11.run(buf127, convolution_64, 150528, grid=grid(150528), stream=stream0)
        del convolution_64
        buf128 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf127, buf128, 384, 392, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf129 = aten.convolution_backward(buf127, mul_216, view_140, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del buf127
        del mul_216
        del view_140
        buf130 = buf129[0]
        buf131 = buf129[1]
        del buf129
        buf135 = buf92; del buf92  # reuse
        buf134 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_13.run(buf131, primals_139, unsqueeze_138, squeeze_93, primals_140, buf135, buf134, 384, 576, grid=grid(384), stream=stream0)
        del primals_139
        del primals_140
        del squeeze_93
        del unsqueeze_138
        buf136 = buf130; del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_11.run(buf136, convolution_63, 150528, grid=grid(150528), stream=stream0)
        del convolution_63
        buf137 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf136, buf137, 384, 392, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf138 = aten.convolution_backward(buf136, mul_212, view_137, [384], [2, 2], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del buf136
        del mul_212
        del view_137
        buf139 = buf138[0]
        buf140 = buf138[1]
        del buf138
        buf144 = buf131; del buf131  # reuse
        buf143 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_13.run(buf140, primals_136, unsqueeze_146, squeeze_91, primals_137, buf144, buf143, 384, 576, grid=grid(384), stream=stream0)
        del primals_136
        del primals_137
        del squeeze_91
        del unsqueeze_146
        buf145 = buf139; del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf145, convolution_62, 602112, grid=grid(602112), stream=stream0)
        del convolution_62
        buf146 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf145, buf146, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf147 = aten.convolution_backward(buf145, mul_205, view_134, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf145
        del mul_205
        del view_134
        buf148 = buf147[0]
        buf149 = buf147[1]
        del buf147
        buf153 = reinterpret_tensor(buf122, (384, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf122  # reuse
        buf152 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_14.run(buf149, primals_133, unsqueeze_154, squeeze_89, primals_134, buf153, buf152, 384, 1536, grid=grid(384), stream=stream0)
        del primals_133
        del primals_134
        del squeeze_89
        del unsqueeze_154
        buf154 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_9.run(buf106, buf154, 1536, 392, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf155 = aten.convolution_backward(buf106, avg_pool2d_2, view_131, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_2
        del buf106
        del view_131
        buf156 = buf155[0]
        buf157 = buf155[1]
        del buf155
        buf161 = empty((1536, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf160 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_20.run(buf157, primals_130, unsqueeze_162, squeeze_87, primals_131, buf161, buf160, 1536, 1536, grid=grid(1536), stream=stream0)
        del buf157
        del primals_130
        del primals_131
        del squeeze_87
        del unsqueeze_162
        buf162 = reinterpret_tensor(buf116, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf116  # reuse
        buf163 = reinterpret_tensor(buf162, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf162  # reuse
        # Source Nodes: [sigmoid_8], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_red_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_sum_21.run(buf163, buf148, buf156, mul_469, convolution_58, convolution_60, 12288, 196, grid=grid(12288), stream=stream0)
        del convolution_58
        buf164 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_5.run(buf163, buf164, 1536, 8, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf165 = aten.convolution_backward(buf163, relu_8, primals_206, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf163
        del primals_206
        buf166 = buf165[0]
        buf167 = buf165[1]
        del buf165
        buf168 = buf166; del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_6.run(buf168, relu_8, 3072, grid=grid(3072), stream=stream0)
        del relu_8
        buf169 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_7.run(buf168, buf169, 384, 8, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf170 = aten.convolution_backward(buf168, mean_8, primals_204, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf168
        del mean_8
        del primals_204
        buf171 = buf170[0]
        buf172 = buf170[1]
        del buf170
        buf173 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_8], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_avg_pool2d_backward_div_mul_sigmoid_22.run(buf148, buf156, mul_469, convolution_60, buf171, buf173, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_60
        buf174 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_23.run(buf173, buf174, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf175 = aten.convolution_backward(buf173, mul_197, view_128, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf173
        del mul_197
        del view_128
        buf176 = buf175[0]
        buf177 = buf175[1]
        del buf175
        buf181 = reinterpret_tensor(buf149, (1536, 384, 1, 1), (384, 1, 1, 1), 0); del buf149  # reuse
        buf180 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_10.run(buf177, primals_127, unsqueeze_170, squeeze_85, primals_128, buf181, buf180, 1536, 384, grid=grid(1536), stream=stream0)
        del primals_127
        del primals_128
        del squeeze_85
        del unsqueeze_170
        buf182 = buf176; del buf176  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act3], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf182, convolution_57, 602112, grid=grid(602112), stream=stream0)
        del convolution_57
        buf183 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf182, buf183, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf184 = aten.convolution_backward(buf182, mul_193, view_125, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del buf182
        del mul_193
        del view_125
        buf185 = buf184[0]
        buf186 = buf184[1]
        del buf184
        buf190 = buf140; del buf140  # reuse
        buf189 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_13.run(buf186, primals_124, unsqueeze_178, squeeze_83, primals_125, buf190, buf189, 384, 576, grid=grid(384), stream=stream0)
        del primals_124
        del primals_125
        del squeeze_83
        del unsqueeze_178
        buf191 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf191, convolution_56, 602112, grid=grid(602112), stream=stream0)
        del convolution_56
        buf192 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf191, buf192, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf193 = aten.convolution_backward(buf191, mul_189, view_122, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del buf191
        del mul_189
        del view_122
        buf194 = buf193[0]
        buf195 = buf193[1]
        del buf193
        buf199 = buf186; del buf186  # reuse
        buf198 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_13.run(buf195, primals_121, unsqueeze_186, squeeze_81, primals_122, buf199, buf198, 384, 576, grid=grid(384), stream=stream0)
        del primals_121
        del primals_122
        del squeeze_81
        del unsqueeze_186
        buf200 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf200, convolution_55, 602112, grid=grid(602112), stream=stream0)
        del convolution_55
        buf201 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf200, buf201, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf202 = aten.convolution_backward(buf200, mul_185, view_119, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf200
        del mul_185
        del view_119
        buf203 = buf202[0]
        buf204 = buf202[1]
        del buf202
        buf208 = reinterpret_tensor(buf177, (384, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf177  # reuse
        buf207 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_14.run(buf204, primals_118, unsqueeze_194, squeeze_79, primals_119, buf208, buf207, 384, 1536, grid=grid(384), stream=stream0)
        del primals_118
        del primals_119
        del squeeze_79
        del unsqueeze_194
        buf209 = buf148; del buf148  # reuse
        buf210 = reinterpret_tensor(buf171, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf171  # reuse
        buf211 = reinterpret_tensor(buf210, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf210  # reuse
        # Source Nodes: [sigmoid_7], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_sum_24.run(buf209, buf211, buf156, mul_469, buf203, mul_528, convolution_52, convolution_54, 12288, 196, grid=grid(12288), stream=stream0)
        del buf156
        del convolution_52
        del mul_469
        del mul_528
        buf212 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_5.run(buf211, buf212, 1536, 8, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf213 = aten.convolution_backward(buf211, relu_7, primals_202, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf211
        del primals_202
        buf214 = buf213[0]
        buf215 = buf213[1]
        del buf213
        buf216 = buf214; del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_6.run(buf216, relu_7, 3072, grid=grid(3072), stream=stream0)
        del relu_7
        buf217 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_7.run(buf216, buf217, 384, 8, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf218 = aten.convolution_backward(buf216, mean_7, primals_200, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf216
        del mean_7
        del primals_200
        buf219 = buf218[0]
        buf220 = buf218[1]
        del buf218
        buf221 = buf203; del buf203  # reuse
        # Source Nodes: [sigmoid_7], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_mul_sigmoid_25.run(buf209, convolution_54, buf219, buf221, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_54
        buf222 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_23.run(buf221, buf222, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf223 = aten.convolution_backward(buf221, mul_177, view_116, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_177
        del view_116
        buf224 = buf223[0]
        buf225 = buf223[1]
        del buf223
        buf229 = reinterpret_tensor(buf204, (1536, 384, 1, 1), (384, 1, 1, 1), 0); del buf204  # reuse
        buf228 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_10.run(buf225, primals_115, unsqueeze_202, squeeze_77, primals_116, buf229, buf228, 1536, 384, grid=grid(1536), stream=stream0)
        del primals_115
        del primals_116
        del squeeze_77
        del unsqueeze_202
        buf230 = buf224; del buf224  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act3], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf230, convolution_51, 602112, grid=grid(602112), stream=stream0)
        del convolution_51
        buf231 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf230, buf231, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf232 = aten.convolution_backward(buf230, mul_173, view_113, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del buf230
        del mul_173
        del view_113
        buf233 = buf232[0]
        buf234 = buf232[1]
        del buf232
        buf238 = buf195; del buf195  # reuse
        buf237 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_13.run(buf234, primals_112, unsqueeze_210, squeeze_75, primals_113, buf238, buf237, 384, 576, grid=grid(384), stream=stream0)
        del primals_112
        del primals_113
        del squeeze_75
        del unsqueeze_210
        buf239 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf239, convolution_50, 602112, grid=grid(602112), stream=stream0)
        del convolution_50
        buf240 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf239, buf240, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf241 = aten.convolution_backward(buf239, mul_169, view_110, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del buf239
        del mul_169
        del view_110
        buf242 = buf241[0]
        buf243 = buf241[1]
        del buf241
        buf247 = buf234; del buf234  # reuse
        buf246 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_13.run(buf243, primals_109, unsqueeze_218, squeeze_73, primals_110, buf247, buf246, 384, 576, grid=grid(384), stream=stream0)
        del primals_109
        del primals_110
        del squeeze_73
        del unsqueeze_218
        buf248 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf248, convolution_49, 602112, grid=grid(602112), stream=stream0)
        del convolution_49
        buf249 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf248, buf249, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf250 = aten.convolution_backward(buf248, mul_165, view_107, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf248
        del mul_165
        del view_107
        buf251 = buf250[0]
        buf252 = buf250[1]
        del buf250
        buf256 = reinterpret_tensor(buf225, (384, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf225  # reuse
        buf255 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_14.run(buf252, primals_106, unsqueeze_226, squeeze_71, primals_107, buf256, buf255, 384, 1536, grid=grid(384), stream=stream0)
        del primals_106
        del primals_107
        del squeeze_71
        del unsqueeze_226
        buf257 = reinterpret_tensor(buf219, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf219  # reuse
        buf258 = reinterpret_tensor(buf257, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf257  # reuse
        # Source Nodes: [sigmoid_6], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_26.run(buf258, buf209, buf251, mul_587, convolution_46, convolution_48, 12288, 196, grid=grid(12288), stream=stream0)
        del convolution_46
        buf259 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_5.run(buf258, buf259, 1536, 8, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf260 = aten.convolution_backward(buf258, relu_6, primals_198, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf258
        del primals_198
        buf261 = buf260[0]
        buf262 = buf260[1]
        del buf260
        buf263 = buf261; del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_6.run(buf263, relu_6, 3072, grid=grid(3072), stream=stream0)
        del relu_6
        buf264 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_7.run(buf263, buf264, 384, 8, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf265 = aten.convolution_backward(buf263, mean_6, primals_196, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf263
        del mean_6
        del primals_196
        buf266 = buf265[0]
        buf267 = buf265[1]
        del buf265
        buf268 = buf221; del buf221  # reuse
        # Source Nodes: [sigmoid_6], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_mul_sigmoid_27.run(buf209, buf251, mul_587, convolution_48, buf266, buf268, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_48
        buf269 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_23.run(buf268, buf269, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf270 = aten.convolution_backward(buf268, mul_157, view_104, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf268
        del mul_157
        del view_104
        buf271 = buf270[0]
        buf272 = buf270[1]
        del buf270
        buf276 = reinterpret_tensor(buf252, (1536, 384, 1, 1), (384, 1, 1, 1), 0); del buf252  # reuse
        buf275 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_10.run(buf272, primals_103, unsqueeze_234, squeeze_69, primals_104, buf276, buf275, 1536, 384, grid=grid(1536), stream=stream0)
        del primals_103
        del primals_104
        del squeeze_69
        del unsqueeze_234
        buf277 = buf271; del buf271  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act3], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf277, convolution_45, 602112, grid=grid(602112), stream=stream0)
        del convolution_45
        buf278 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf277, buf278, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf279 = aten.convolution_backward(buf277, mul_153, view_101, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del buf277
        del mul_153
        del view_101
        buf280 = buf279[0]
        buf281 = buf279[1]
        del buf279
        buf285 = buf243; del buf243  # reuse
        buf284 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_13.run(buf281, primals_100, unsqueeze_242, squeeze_67, primals_101, buf285, buf284, 384, 576, grid=grid(384), stream=stream0)
        del primals_100
        del primals_101
        del squeeze_67
        del unsqueeze_242
        buf286 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf286, convolution_44, 602112, grid=grid(602112), stream=stream0)
        del convolution_44
        buf287 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf286, buf287, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf288 = aten.convolution_backward(buf286, mul_149, view_98, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del buf286
        del mul_149
        del view_98
        buf289 = buf288[0]
        buf290 = buf288[1]
        del buf288
        buf294 = buf281; del buf281  # reuse
        buf293 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_13.run(buf290, primals_97, unsqueeze_250, squeeze_65, primals_98, buf294, buf293, 384, 576, grid=grid(384), stream=stream0)
        del primals_97
        del primals_98
        del squeeze_65
        del unsqueeze_250
        buf295 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf295, convolution_43, 602112, grid=grid(602112), stream=stream0)
        del convolution_43
        buf296 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf295, buf296, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf297 = aten.convolution_backward(buf295, mul_145, view_95, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf295
        del mul_145
        del view_95
        buf298 = buf297[0]
        buf299 = buf297[1]
        del buf297
        buf303 = reinterpret_tensor(buf272, (384, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf272  # reuse
        buf302 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_14.run(buf299, primals_94, unsqueeze_258, squeeze_63, primals_95, buf303, buf302, 384, 1536, grid=grid(384), stream=stream0)
        del primals_94
        del primals_95
        del squeeze_63
        del unsqueeze_258
        buf304 = buf209; del buf209  # reuse
        buf305 = reinterpret_tensor(buf266, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf266  # reuse
        buf306 = reinterpret_tensor(buf305, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf305  # reuse
        # Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_28.run(buf304, buf306, buf251, mul_587, buf298, mul_646, convolution_40, convolution_42, 12288, 196, grid=grid(12288), stream=stream0)
        del buf251
        del convolution_40
        del mul_587
        del mul_646
        buf307 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_5.run(buf306, buf307, 1536, 8, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf308 = aten.convolution_backward(buf306, relu_5, primals_194, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf306
        del primals_194
        buf309 = buf308[0]
        buf310 = buf308[1]
        del buf308
        buf311 = buf309; del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_6.run(buf311, relu_5, 3072, grid=grid(3072), stream=stream0)
        del relu_5
        buf312 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_7.run(buf311, buf312, 384, 8, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf313 = aten.convolution_backward(buf311, mean_5, primals_192, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf311
        del mean_5
        del primals_192
        buf314 = buf313[0]
        buf315 = buf313[1]
        del buf313
        buf316 = buf298; del buf298  # reuse
        # Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_mul_sigmoid_25.run(buf304, convolution_42, buf314, buf316, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_42
        buf317 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_23.run(buf316, buf317, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf318 = aten.convolution_backward(buf316, mul_137, view_92, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_137
        del view_92
        buf319 = buf318[0]
        buf320 = buf318[1]
        del buf318
        buf324 = reinterpret_tensor(buf299, (1536, 384, 1, 1), (384, 1, 1, 1), 0); del buf299  # reuse
        buf323 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_10.run(buf320, primals_91, unsqueeze_266, squeeze_61, primals_92, buf324, buf323, 1536, 384, grid=grid(1536), stream=stream0)
        del primals_91
        del primals_92
        del squeeze_61
        del unsqueeze_266
        buf325 = buf319; del buf319  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act3], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf325, convolution_39, 602112, grid=grid(602112), stream=stream0)
        del convolution_39
        buf326 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf325, buf326, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf327 = aten.convolution_backward(buf325, mul_133, view_89, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del buf325
        del mul_133
        del view_89
        buf328 = buf327[0]
        buf329 = buf327[1]
        del buf327
        buf333 = buf290; del buf290  # reuse
        buf332 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_13.run(buf329, primals_88, unsqueeze_274, squeeze_59, primals_89, buf333, buf332, 384, 576, grid=grid(384), stream=stream0)
        del primals_88
        del primals_89
        del squeeze_59
        del unsqueeze_274
        buf334 = buf328; del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf334, convolution_38, 602112, grid=grid(602112), stream=stream0)
        del convolution_38
        buf335 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf334, buf335, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf336 = aten.convolution_backward(buf334, mul_129, view_86, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del buf334
        del mul_129
        del view_86
        buf337 = buf336[0]
        buf338 = buf336[1]
        del buf336
        buf342 = buf329; del buf329  # reuse
        buf341 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_13.run(buf338, primals_85, unsqueeze_282, squeeze_57, primals_86, buf342, buf341, 384, 576, grid=grid(384), stream=stream0)
        del primals_85
        del primals_86
        del squeeze_57
        del unsqueeze_282
        buf343 = buf337; del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf343, convolution_37, 602112, grid=grid(602112), stream=stream0)
        del convolution_37
        buf344 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf343, buf344, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf345 = aten.convolution_backward(buf343, mul_125, view_83, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf343
        del mul_125
        del view_83
        buf346 = buf345[0]
        buf347 = buf345[1]
        del buf345
        buf351 = reinterpret_tensor(buf320, (384, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf320  # reuse
        buf350 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_14.run(buf347, primals_82, unsqueeze_290, squeeze_55, primals_83, buf351, buf350, 384, 1536, grid=grid(384), stream=stream0)
        del primals_82
        del primals_83
        del squeeze_55
        del unsqueeze_290
        buf352 = reinterpret_tensor(buf314, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf314  # reuse
        buf353 = reinterpret_tensor(buf352, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf352  # reuse
        # Source Nodes: [sigmoid_4], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_29.run(buf353, buf304, buf346, mul_705, convolution_34, convolution_36, 12288, 196, grid=grid(12288), stream=stream0)
        del convolution_34
        buf354 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_5.run(buf353, buf354, 1536, 8, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf355 = aten.convolution_backward(buf353, relu_4, primals_190, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf353
        del primals_190
        buf356 = buf355[0]
        buf357 = buf355[1]
        del buf355
        buf358 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_6.run(buf358, relu_4, 3072, grid=grid(3072), stream=stream0)
        del relu_4
        buf359 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_7.run(buf358, buf359, 384, 8, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf360 = aten.convolution_backward(buf358, mean_4, primals_188, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf358
        del mean_4
        del primals_188
        buf361 = buf360[0]
        buf362 = buf360[1]
        del buf360
        buf363 = buf316; del buf316  # reuse
        # Source Nodes: [sigmoid_4], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_mul_sigmoid_30.run(buf304, buf346, mul_705, convolution_36, buf361, buf363, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_36
        buf364 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_23.run(buf363, buf364, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf365 = aten.convolution_backward(buf363, mul_117, view_80, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf363
        del mul_117
        del view_80
        buf366 = buf365[0]
        buf367 = buf365[1]
        del buf365
        buf371 = reinterpret_tensor(buf347, (1536, 384, 1, 1), (384, 1, 1, 1), 0); del buf347  # reuse
        buf370 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_10.run(buf367, primals_79, unsqueeze_298, squeeze_53, primals_80, buf371, buf370, 1536, 384, grid=grid(1536), stream=stream0)
        del primals_79
        del primals_80
        del squeeze_53
        del unsqueeze_298
        buf372 = buf366; del buf366  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act3], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf372, convolution_33, 602112, grid=grid(602112), stream=stream0)
        del convolution_33
        buf373 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf372, buf373, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf374 = aten.convolution_backward(buf372, mul_113, view_77, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del buf372
        del mul_113
        del view_77
        buf375 = buf374[0]
        buf376 = buf374[1]
        del buf374
        buf380 = buf338; del buf338  # reuse
        buf379 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_13.run(buf376, primals_76, unsqueeze_306, squeeze_51, primals_77, buf380, buf379, 384, 576, grid=grid(384), stream=stream0)
        del primals_76
        del primals_77
        del squeeze_51
        del unsqueeze_306
        buf381 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf381, convolution_32, 602112, grid=grid(602112), stream=stream0)
        del convolution_32
        buf382 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf381, buf382, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf383 = aten.convolution_backward(buf381, mul_109, view_74, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del buf381
        del mul_109
        del view_74
        buf384 = buf383[0]
        buf385 = buf383[1]
        del buf383
        buf389 = buf376; del buf376  # reuse
        buf388 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_13.run(buf385, primals_73, unsqueeze_314, squeeze_49, primals_74, buf389, buf388, 384, 576, grid=grid(384), stream=stream0)
        del primals_73
        del primals_74
        del squeeze_49
        del unsqueeze_314
        buf390 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf390, convolution_31, 602112, grid=grid(602112), stream=stream0)
        del convolution_31
        buf391 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf390, buf391, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf392 = aten.convolution_backward(buf390, mul_105, view_71, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf390
        del mul_105
        del view_71
        buf393 = buf392[0]
        buf394 = buf392[1]
        del buf392
        buf398 = reinterpret_tensor(buf367, (384, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf367  # reuse
        buf397 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_14.run(buf394, primals_70, unsqueeze_322, squeeze_47, primals_71, buf398, buf397, 384, 1536, grid=grid(384), stream=stream0)
        del primals_70
        del primals_71
        del squeeze_47
        del unsqueeze_322
        buf399 = buf304; del buf304  # reuse
        buf400 = reinterpret_tensor(buf361, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf361  # reuse
        buf401 = reinterpret_tensor(buf400, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf400  # reuse
        # Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_31.run(buf399, buf401, buf346, mul_705, buf393, mul_764, convolution_28, convolution_30, 12288, 196, grid=grid(12288), stream=stream0)
        del buf346
        del convolution_28
        del mul_705
        del mul_764
        buf402 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_5.run(buf401, buf402, 1536, 8, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf403 = aten.convolution_backward(buf401, relu_3, primals_186, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf401
        del primals_186
        buf404 = buf403[0]
        buf405 = buf403[1]
        del buf403
        buf406 = buf404; del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_6.run(buf406, relu_3, 3072, grid=grid(3072), stream=stream0)
        del relu_3
        buf407 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_7.run(buf406, buf407, 384, 8, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf408 = aten.convolution_backward(buf406, mean_3, primals_184, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf406
        del mean_3
        del primals_184
        buf409 = buf408[0]
        buf410 = buf408[1]
        del buf408
        buf411 = buf393; del buf393  # reuse
        # Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_mul_sigmoid_25.run(buf399, convolution_30, buf409, buf411, 2408448, grid=grid(2408448), stream=stream0)
        del buf409
        del convolution_30
        buf412 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_23.run(buf411, buf412, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf413 = aten.convolution_backward(buf411, mul_97, view_68, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf411
        del mul_97
        del view_68
        buf414 = buf413[0]
        buf415 = buf413[1]
        del buf413
        buf419 = reinterpret_tensor(buf394, (1536, 384, 1, 1), (384, 1, 1, 1), 0); del buf394  # reuse
        buf418 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_10.run(buf415, primals_67, unsqueeze_330, squeeze_45, primals_68, buf419, buf418, 1536, 384, grid=grid(1536), stream=stream0)
        del buf415
        del primals_67
        del primals_68
        del squeeze_45
        del unsqueeze_330
        buf420 = buf414; del buf414  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act3], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf420, convolution_27, 602112, grid=grid(602112), stream=stream0)
        del convolution_27
        buf421 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf420, buf421, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf422 = aten.convolution_backward(buf420, mul_93, view_65, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del buf420
        del mul_93
        del view_65
        buf423 = buf422[0]
        buf424 = buf422[1]
        del buf422
        buf428 = buf385; del buf385  # reuse
        buf427 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_13.run(buf424, primals_64, unsqueeze_338, squeeze_43, primals_65, buf428, buf427, 384, 576, grid=grid(384), stream=stream0)
        del primals_64
        del primals_65
        del squeeze_43
        del unsqueeze_338
        buf429 = buf423; del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf429, convolution_26, 602112, grid=grid(602112), stream=stream0)
        del convolution_26
        buf430 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf429, buf430, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf431 = aten.convolution_backward(buf429, mul_89, view_62, [384], [2, 2], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del buf429
        del mul_89
        del view_62
        buf432 = buf431[0]
        buf433 = buf431[1]
        del buf431
        buf437 = buf424; del buf424  # reuse
        buf436 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_13.run(buf433, primals_61, unsqueeze_346, squeeze_41, primals_62, buf437, buf436, 384, 576, grid=grid(384), stream=stream0)
        del buf433
        del primals_61
        del primals_62
        del squeeze_41
        del unsqueeze_346
        buf438 = buf432; del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_32.run(buf438, convolution_25, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_25
        buf439 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf438, buf439, 384, 6272, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf440 = aten.convolution_backward(buf438, mul_82, view_59, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf438
        del mul_82
        del view_59
        buf441 = buf440[0]
        buf442 = buf440[1]
        del buf440
        buf446 = empty((384, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf445 = empty((384, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_34.run(buf442, primals_58, unsqueeze_354, squeeze_39, primals_59, buf446, buf445, 384, 512, grid=grid(384), stream=stream0)
        del buf442
        del primals_58
        del primals_59
        del squeeze_39
        del unsqueeze_354
        buf447 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_23.run(buf399, buf447, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf448 = aten.convolution_backward(buf399, avg_pool2d_1, view_56, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_1
        del buf399
        del view_56
        buf449 = buf448[0]
        buf450 = buf448[1]
        del buf448
        buf454 = empty((1536, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf453 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_35.run(buf450, primals_55, unsqueeze_362, squeeze_37, primals_56, buf454, buf453, 1536, 512, grid=grid(1536), stream=stream0)
        del buf450
        del primals_55
        del primals_56
        del squeeze_37
        del unsqueeze_362
        buf455 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf456 = reinterpret_tensor(buf455, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf455  # reuse
        # Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_sum_36.run(buf456, buf441, buf449, mul_833, convolution_21, convolution_23, 4096, 784, grid=grid(4096), stream=stream0)
        del convolution_21
        buf457 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_37.run(buf456, buf457, 512, 8, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf458 = aten.convolution_backward(buf456, relu_2, primals_182, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf456
        del primals_182
        buf459 = buf458[0]
        buf460 = buf458[1]
        del buf458
        buf461 = buf459; del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_38.run(buf461, relu_2, 1024, grid=grid(1024), stream=stream0)
        del relu_2
        buf462 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_39.run(buf461, buf462, 128, 8, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf463 = aten.convolution_backward(buf461, mean_2, primals_180, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf461
        del mean_2
        del primals_180
        buf464 = buf463[0]
        buf465 = buf463[1]
        del buf463
        buf466 = empty((8, 512, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_avg_pool2d_backward_div_mul_sigmoid_40.run(buf441, buf449, mul_833, convolution_23, buf464, buf466, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_23
        buf467 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_41.run(buf466, buf467, 512, 6272, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf468 = aten.convolution_backward(buf466, mul_74, view_53, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf466
        del mul_74
        del view_53
        buf469 = buf468[0]
        buf470 = buf468[1]
        del buf468
        buf474 = empty((512, 128, 1, 1), device='cuda', dtype=torch.float32)
        buf473 = empty((512, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_42.run(buf470, primals_52, unsqueeze_370, squeeze_35, primals_53, buf474, buf473, 512, 128, grid=grid(512), stream=stream0)
        del primals_52
        del primals_53
        del squeeze_35
        del unsqueeze_370
        buf475 = buf469; del buf469  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act3], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_43.run(buf475, convolution_20, 802816, grid=grid(802816), stream=stream0)
        del convolution_20
        buf476 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_44.run(buf475, buf476, 128, 6272, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf477 = aten.convolution_backward(buf475, mul_70, view_50, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del buf475
        del mul_70
        del view_50
        buf478 = buf477[0]
        buf479 = buf477[1]
        del buf477
        buf483 = empty((128, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf482 = empty((128, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_45.run(buf479, primals_49, unsqueeze_378, squeeze_33, primals_50, buf483, buf482, 128, 576, grid=grid(128), stream=stream0)
        del primals_49
        del primals_50
        del squeeze_33
        del unsqueeze_378
        buf484 = buf478; del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_43.run(buf484, convolution_19, 802816, grid=grid(802816), stream=stream0)
        del convolution_19
        buf485 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_44.run(buf484, buf485, 128, 6272, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf486 = aten.convolution_backward(buf484, mul_66, view_47, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del buf484
        del mul_66
        del view_47
        buf487 = buf486[0]
        buf488 = buf486[1]
        del buf486
        buf492 = buf479; del buf479  # reuse
        buf491 = empty((128, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_45.run(buf488, primals_46, unsqueeze_386, squeeze_31, primals_47, buf492, buf491, 128, 576, grid=grid(128), stream=stream0)
        del primals_46
        del primals_47
        del squeeze_31
        del unsqueeze_386
        buf493 = buf487; del buf487  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_43.run(buf493, convolution_18, 802816, grid=grid(802816), stream=stream0)
        del convolution_18
        buf494 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_44.run(buf493, buf494, 128, 6272, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf495 = aten.convolution_backward(buf493, mul_62, view_44, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf493
        del mul_62
        del view_44
        buf496 = buf495[0]
        buf497 = buf495[1]
        del buf495
        buf501 = reinterpret_tensor(buf470, (128, 512, 1, 1), (512, 1, 1, 1), 0); del buf470  # reuse
        buf500 = empty((128, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_46.run(buf497, primals_43, unsqueeze_394, squeeze_29, primals_44, buf501, buf500, 128, 512, grid=grid(128), stream=stream0)
        del primals_43
        del primals_44
        del squeeze_29
        del unsqueeze_394
        buf502 = buf441; del buf441  # reuse
        buf503 = reinterpret_tensor(buf464, (8, 512, 1, 1), (512, 1, 4096, 4096), 0); del buf464  # reuse
        buf504 = reinterpret_tensor(buf503, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf503  # reuse
        # Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_sum_47.run(buf502, buf504, buf449, mul_833, buf496, mul_892, convolution_15, convolution_17, 4096, 784, grid=grid(4096), stream=stream0)
        del buf449
        del convolution_15
        del mul_833
        del mul_892
        buf505 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_37.run(buf504, buf505, 512, 8, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf506 = aten.convolution_backward(buf504, relu_1, primals_178, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf504
        del primals_178
        buf507 = buf506[0]
        buf508 = buf506[1]
        del buf506
        buf509 = buf507; del buf507  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_38.run(buf509, relu_1, 1024, grid=grid(1024), stream=stream0)
        del relu_1
        buf510 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_39.run(buf509, buf510, 128, 8, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf511 = aten.convolution_backward(buf509, mean_1, primals_176, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf509
        del mean_1
        del primals_176
        buf512 = buf511[0]
        buf513 = buf511[1]
        del buf511
        buf514 = buf496; del buf496  # reuse
        # Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_mul_sigmoid_48.run(buf502, convolution_17, buf512, buf514, 3211264, grid=grid(3211264), stream=stream0)
        del buf512
        del convolution_17
        buf515 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_41.run(buf514, buf515, 512, 6272, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf516 = aten.convolution_backward(buf514, mul_54, view_41, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf514
        del mul_54
        del view_41
        buf517 = buf516[0]
        buf518 = buf516[1]
        del buf516
        buf522 = reinterpret_tensor(buf497, (512, 128, 1, 1), (128, 1, 1, 1), 0); del buf497  # reuse
        buf521 = empty((512, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_42.run(buf518, primals_40, unsqueeze_402, squeeze_27, primals_41, buf522, buf521, 512, 128, grid=grid(512), stream=stream0)
        del buf518
        del primals_40
        del primals_41
        del squeeze_27
        del unsqueeze_402
        buf523 = buf517; del buf517  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act3], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_43.run(buf523, convolution_14, 802816, grid=grid(802816), stream=stream0)
        del convolution_14
        buf524 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_44.run(buf523, buf524, 128, 6272, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf525 = aten.convolution_backward(buf523, mul_50, view_38, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del buf523
        del mul_50
        del view_38
        buf526 = buf525[0]
        buf527 = buf525[1]
        del buf525
        buf531 = buf488; del buf488  # reuse
        buf530 = empty((128, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_45.run(buf527, primals_37, unsqueeze_410, squeeze_25, primals_38, buf531, buf530, 128, 576, grid=grid(128), stream=stream0)
        del primals_37
        del primals_38
        del squeeze_25
        del unsqueeze_410
        buf532 = buf526; del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_43.run(buf532, convolution_13, 802816, grid=grid(802816), stream=stream0)
        del convolution_13
        buf533 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_44.run(buf532, buf533, 128, 6272, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf534 = aten.convolution_backward(buf532, mul_46, view_35, [128], [2, 2], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del buf532
        del mul_46
        del view_35
        buf535 = buf534[0]
        buf536 = buf534[1]
        del buf534
        buf540 = buf527; del buf527  # reuse
        buf539 = empty((128, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_45.run(buf536, primals_34, unsqueeze_418, squeeze_23, primals_35, buf540, buf539, 128, 576, grid=grid(128), stream=stream0)
        del primals_34
        del primals_35
        del squeeze_23
        del unsqueeze_418
        buf541 = buf535; del buf535  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_49.run(buf541, convolution_12, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_12
        buf542 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_50.run(buf541, buf542, 512, 6272, grid=grid(512), stream=stream0)
        buf543 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_51.run(buf542, buf543, 128, 4, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf544 = aten.convolution_backward(buf541, mul_39, view_32, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf541
        del mul_39
        del view_32
        buf545 = buf544[0]
        buf546 = buf544[1]
        del buf544
        buf550 = empty((128, 256, 1, 1), device='cuda', dtype=torch.float32)
        buf549 = empty((128, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_52.run(buf546, primals_31, unsqueeze_426, squeeze_21, primals_32, buf550, buf549, 128, 256, grid=grid(128), stream=stream0)
        del primals_31
        del primals_32
        del squeeze_21
        del unsqueeze_426
        buf551 = reinterpret_tensor(buf542, (512, ), (1, ), 0); del buf542  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_41.run(buf502, buf551, 512, 6272, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf552 = aten.convolution_backward(buf502, avg_pool2d, view_29, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d
        del buf502
        del view_29
        buf553 = buf552[0]
        buf554 = buf552[1]
        del buf552
        buf558 = empty((512, 256, 1, 1), device='cuda', dtype=torch.float32)
        buf557 = empty((512, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_53.run(buf554, primals_28, unsqueeze_434, squeeze_19, primals_29, buf558, buf557, 512, 256, grid=grid(512), stream=stream0)
        del buf554
        del primals_28
        del primals_29
        del squeeze_19
        del unsqueeze_434
        buf559 = buf545; del buf545  # reuse
        buf560 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf561 = reinterpret_tensor(buf560, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf560  # reuse
        # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_red_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_sum_54.run(buf559, buf561, buf553, mul_961, convolution_8, convolution_10, 2048, 3136, grid=grid(2048), stream=stream0)
        del buf553
        del convolution_8
        del mul_961
        buf562 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_55.run(buf561, buf562, 256, 8, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf563 = aten.convolution_backward(buf561, relu, primals_174, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf561
        del primals_174
        buf564 = buf563[0]
        buf565 = buf563[1]
        del buf563
        buf566 = buf564; del buf564  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_56.run(buf566, relu, 512, grid=grid(512), stream=stream0)
        del relu
        buf567 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_57.run(buf566, buf567, 64, 8, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf568 = aten.convolution_backward(buf566, mean, primals_172, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean
        del primals_172
        buf569 = buf568[0]
        buf570 = buf568[1]
        del buf568
        buf571 = empty((8, 256, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_mul_sigmoid_58.run(buf559, convolution_10, buf569, buf571, 6422528, grid=grid(6422528), stream=stream0)
        del buf569
        del convolution_10
        buf572 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_59.run(buf571, buf572, 256, 25088, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf573 = aten.convolution_backward(buf571, mul_31, view_26, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf571
        del mul_31
        del view_26
        buf574 = buf573[0]
        buf575 = buf573[1]
        del buf573
        buf579 = empty((256, 64, 1, 1), device='cuda', dtype=torch.float32)
        buf578 = empty((256, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_60.run(buf575, primals_25, unsqueeze_442, squeeze_17, primals_26, buf579, buf578, 256, 64, grid=grid(256), stream=stream0)
        del buf575
        del primals_25
        del primals_26
        del squeeze_17
        del unsqueeze_442
        buf580 = buf574; del buf574  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act3], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_61.run(buf580, convolution_7, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_7
        buf581 = empty_strided((64, 4), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_62.run(buf580, buf581, 256, 6272, grid=grid(256), stream=stream0)
        buf582 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_63.run(buf581, buf582, 64, 4, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf583 = aten.convolution_backward(buf580, mul_27, view_23, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf580
        del mul_27
        del view_23
        buf584 = buf583[0]
        buf585 = buf583[1]
        del buf583
        buf589 = empty((64, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf588 = empty((64, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_64.run(buf585, primals_22, unsqueeze_450, squeeze_15, primals_23, buf589, buf588, 64, 576, grid=grid(64), stream=stream0)
        del primals_22
        del primals_23
        del squeeze_15
        del unsqueeze_450
        buf590 = buf584; del buf584  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_61.run(buf590, convolution_6, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_6
        buf591 = buf581; del buf581  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_62.run(buf590, buf591, 256, 6272, grid=grid(256), stream=stream0)
        buf592 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_63.run(buf591, buf592, 64, 4, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf593 = aten.convolution_backward(buf590, mul_23, view_20, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf590
        del mul_23
        del view_20
        buf594 = buf593[0]
        buf595 = buf593[1]
        del buf593
        buf599 = buf585; del buf585  # reuse
        buf598 = empty((64, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_64.run(buf595, primals_19, unsqueeze_458, squeeze_13, primals_20, buf599, buf598, 64, 576, grid=grid(64), stream=stream0)
        del buf595
        del primals_19
        del primals_20
        del squeeze_13
        del unsqueeze_458
        buf600 = buf594; del buf594  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_61.run(buf600, convolution_5, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_5
        buf601 = buf591; del buf591  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_62.run(buf600, buf601, 256, 6272, grid=grid(256), stream=stream0)
        buf602 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_63.run(buf601, buf602, 64, 4, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf603 = aten.convolution_backward(buf600, mul_16, view_17, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf600
        del view_17
        buf604 = buf603[0]
        buf605 = buf603[1]
        del buf603
        buf609 = empty((64, 128, 1, 1), device='cuda', dtype=torch.float32)
        buf608 = empty((64, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_65.run(buf605, primals_16, unsqueeze_466, squeeze_11, primals_17, buf609, buf608, 64, 128, grid=grid(64), stream=stream0)
        del buf605
        del primals_16
        del primals_17
        del squeeze_11
        del unsqueeze_466
        buf610 = reinterpret_tensor(buf601, (256, ), (1, ), 0); del buf601  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_59.run(buf559, buf610, 256, 25088, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf611 = aten.convolution_backward(buf559, mul_16, view_14, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf559
        del mul_16
        del view_14
        buf612 = buf611[0]
        buf613 = buf611[1]
        del buf611
        buf617 = reinterpret_tensor(buf546, (256, 128, 1, 1), (128, 1, 1, 1), 0); del buf546  # reuse
        buf616 = empty((256, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_66.run(buf613, primals_13, unsqueeze_474, squeeze_9, primals_14, buf617, buf616, 256, 128, grid=grid(256), stream=stream0)
        del buf613
        del primals_13
        del primals_14
        del squeeze_9
        del unsqueeze_474
        buf618 = buf604; del buf604  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act1], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_67.run(buf618, buf612, convolution_3, 3211264, grid=grid(3211264), stream=stream0)
        del buf612
        del convolution_3
        buf619 = reinterpret_tensor(buf566, (128, 4), (1, 128), 0); del buf566  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_50.run(buf618, buf619, 512, 6272, grid=grid(512), stream=stream0)
        buf620 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_51.run(buf619, buf620, 128, 4, grid=grid(128), stream=stream0)
        del buf619
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf621 = aten.convolution_backward(buf618, mul_11, view_11, [128], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf618
        del mul_11
        del view_11
        buf622 = buf621[0]
        buf623 = buf621[1]
        del buf621
        buf627 = buf536; del buf536  # reuse
        buf626 = empty((128, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_45.run(buf623, primals_10, unsqueeze_482, squeeze_7, primals_11, buf627, buf626, 128, 576, grid=grid(128), stream=stream0)
        del buf623
        del primals_10
        del primals_11
        del squeeze_7
        del unsqueeze_482
        buf628 = buf622; del buf622  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_68.run(buf628, convolution_2, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_2
        buf629 = empty((64, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_69.run(buf628, buf629, 832, 7720, grid=grid(832), stream=stream0)
        buf630 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_70.run(buf629, buf630, 64, 13, grid=grid(64), stream=stream0)
        del buf629
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf631 = aten.convolution_backward(buf628, mul_7, view_8, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf628
        del mul_7
        del view_8
        buf632 = buf631[0]
        buf633 = buf631[1]
        del buf631
        buf637 = reinterpret_tensor(buf0, (64, 32, 3, 3), (288, 9, 3, 1), 0); del buf0  # reuse
        buf636 = empty((64, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_71.run(buf633, primals_7, unsqueeze_490, squeeze_5, primals_8, buf637, buf636, 64, 288, grid=grid(64), stream=stream0)
        del buf633
        del primals_7
        del primals_8
        del squeeze_5
        del unsqueeze_490
        buf638 = buf632; del buf632  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_49.run(buf638, convolution_1, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_1
        buf639 = empty((32, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_72.run(buf638, buf639, 416, 7720, grid=grid(416), stream=stream0)
        buf640 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_73.run(buf639, buf640, 32, 13, grid=grid(32), stream=stream0)
        del buf639
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf641 = aten.convolution_backward(buf638, mul_3, view_5, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf638
        del mul_3
        del view_5
        buf642 = buf641[0]
        buf643 = buf641[1]
        del buf641
        buf647 = empty((32, 16, 3, 3), device='cuda', dtype=torch.float32)
        buf646 = empty((32, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_74.run(buf643, primals_4, unsqueeze_498, squeeze_3, primals_5, buf647, buf646, 32, 144, grid=grid(32), stream=stream0)
        del buf643
        del primals_4
        del primals_5
        del squeeze_3
        del unsqueeze_498
        buf648 = buf642; del buf642  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_61.run(buf648, convolution, 1605632, grid=grid(1605632), stream=stream0)
        del convolution
        buf649 = empty((16, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_75.run(buf648, buf649, 208, 7720, grid=grid(208), stream=stream0)
        buf650 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_76.run(buf649, buf650, 16, 13, grid=grid(16), stream=stream0)
        del buf649
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf651 = aten.convolution_backward(buf648, primals_222, view_2, [16], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf648
        del primals_222
        del view_2
        buf652 = buf651[1]
        del buf651
        buf656 = empty((16, 3, 3, 3), device='cuda', dtype=torch.float32)
        buf655 = empty((16, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_77.run(buf652, primals_1, unsqueeze_506, squeeze_1, primals_2, buf656, buf655, 16, 27, grid=grid(16), stream=stream0)
        del buf652
        del primals_1
        del primals_2
        del squeeze_1
        del unsqueeze_506
        return (buf656, buf655, buf650, buf647, buf646, buf640, buf637, buf636, buf630, buf627, buf626, buf620, buf617, buf616, buf610, buf609, buf608, buf602, buf599, buf598, buf592, buf589, buf588, buf582, buf579, buf578, buf572, buf558, buf557, buf551, buf550, buf549, buf543, buf540, buf539, buf533, buf531, buf530, buf524, buf522, buf521, buf515, buf501, buf500, buf494, buf492, buf491, buf485, buf483, buf482, buf476, buf474, buf473, buf467, buf454, buf453, buf447, buf446, buf445, buf439, buf437, buf436, buf430, buf428, buf427, buf421, buf419, buf418, buf412, buf398, buf397, buf391, buf389, buf388, buf382, buf380, buf379, buf373, buf371, buf370, buf364, buf351, buf350, buf344, buf342, buf341, buf335, buf333, buf332, buf326, buf324, buf323, buf317, buf303, buf302, buf296, buf294, buf293, buf287, buf285, buf284, buf278, buf276, buf275, buf269, buf256, buf255, buf249, buf247, buf246, buf240, buf238, buf237, buf231, buf229, buf228, buf222, buf208, buf207, buf201, buf199, buf198, buf192, buf190, buf189, buf183, buf181, buf180, buf174, buf161, buf160, buf154, buf153, buf152, buf146, buf144, buf143, buf137, buf135, buf134, buf128, buf126, buf125, buf119, buf105, buf104, buf98, buf96, buf95, buf89, buf87, buf86, buf80, buf78, buf77, buf71, buf58, buf57, buf51, buf49, buf48, buf42, buf40, buf39, buf33, buf31, buf30, buf24, buf11, buf10, buf4, buf570, buf567, buf565, buf562, buf513, buf510, buf508, buf505, buf465, buf462, buf460, buf457, buf410, buf407, buf405, buf402, buf362, buf359, buf357, buf354, buf315, buf312, buf310, buf307, buf267, buf264, buf262, buf259, buf220, buf217, buf215, buf212, buf172, buf169, buf167, buf164, buf117, buf114, buf112, buf109, buf69, buf66, buf64, buf61, buf22, buf19, buf17, buf14, reinterpret_tensor(buf1, (1000, 2304), (2304, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((384, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1536, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((2304, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((2304, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_2 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 16, 112, 112), (200704, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    mul_3 = rand_strided((8, 16, 112, 112), (200704, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    squeeze_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_5 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 112, 112), (401408, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    mul_7 = rand_strided((8, 32, 112, 112), (401408, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    squeeze_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_8 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    mul_11 = rand_strided((8, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_11 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    mul_16 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_9 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_14 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    squeeze_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_17 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    mul_23 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    mul_27 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    mul_31 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_26 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_39 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    avg_pool2d = rand_strided((8, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_29 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    squeeze_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    mul_46 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_35 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 128, 28, 28), (100352, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mul_50 = rand_strided((8, 128, 28, 28), (100352, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_38 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 128, 28, 28), (100352, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mul_54 = rand_strided((8, 128, 28, 28), (100352, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_27 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_41 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mean_1 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_62 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 128, 28, 28), (100352, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mul_66 = rand_strided((8, 128, 28, 28), (100352, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_47 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 128, 28, 28), (100352, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mul_70 = rand_strided((8, 128, 28, 28), (100352, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_50 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 128, 28, 28), (100352, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mul_74 = rand_strided((8, 128, 28, 28), (100352, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_35 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_53 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mean_2 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_82 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    avg_pool2d_1 = rand_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_56 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    squeeze_39 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_59 = rand_strided((384, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mul_89 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_41 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_62 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mul_93 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_65 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mul_97 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_45 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_68 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_3 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((8, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_105 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_47 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_71 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mul_109 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_74 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mul_113 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_51 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_77 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mul_117 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_53 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_80 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_4 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((8, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_125 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_83 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mul_129 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_57 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mul_133 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_59 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_89 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mul_137 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_92 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_5 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((8, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_145 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_63 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_95 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mul_149 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_65 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_98 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mul_153 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_101 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mul_157 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_69 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_6 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((8, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_165 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_71 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_107 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mul_169 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mul_173 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_75 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_113 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mul_177 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_77 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_116 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_7 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((8, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_185 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_119 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mul_189 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_81 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_122 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mul_193 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_83 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_125 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_57 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mul_197 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_128 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_58 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_8 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((8, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_205 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    avg_pool2d_2 = rand_strided((8, 1536, 7, 7), (75264, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_87 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_131 = rand_strided((1536, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    squeeze_89 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_134 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_62 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mul_212 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_137 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_63 = rand_strided((8, 384, 7, 7), (18816, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mul_216 = rand_strided((8, 384, 7, 7), (18816, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_93 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_140 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_64 = rand_strided((8, 384, 7, 7), (18816, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mul_220 = rand_strided((8, 384, 7, 7), (18816, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_95 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_143 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_65 = rand_strided((8, 1536, 7, 7), (75264, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mean_9 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((8, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_67 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_228 = rand_strided((8, 1536, 7, 7), (75264, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_146 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_68 = rand_strided((8, 384, 7, 7), (18816, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mul_232 = rand_strided((8, 384, 7, 7), (18816, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_99 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_149 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_69 = rand_strided((8, 384, 7, 7), (18816, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mul_236 = rand_strided((8, 384, 7, 7), (18816, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_101 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_70 = rand_strided((8, 384, 7, 7), (18816, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mul_240 = rand_strided((8, 384, 7, 7), (18816, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_155 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_71 = rand_strided((8, 1536, 7, 7), (75264, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mean_10 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((8, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_73 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_248 = rand_strided((8, 1536, 7, 7), (75264, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_105 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_158 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_74 = rand_strided((8, 384, 7, 7), (18816, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mul_252 = rand_strided((8, 384, 7, 7), (18816, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_107 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_161 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_75 = rand_strided((8, 384, 7, 7), (18816, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mul_256 = rand_strided((8, 384, 7, 7), (18816, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_164 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_76 = rand_strided((8, 384, 7, 7), (18816, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mul_260 = rand_strided((8, 384, 7, 7), (18816, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_111 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_167 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_77 = rand_strided((8, 1536, 7, 7), (75264, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mean_11 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((8, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_79 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    add_67 = rand_strided((8, 1536, 7, 7), (75264, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_113 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_170 = rand_strided((2304, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_80 = rand_strided((8, 2304, 7, 7), (112896, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    clone_28 = rand_strided((8, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_58 = rand_strided((1, 2304, 1), (2304, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_66 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_74 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_82 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_90 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_341 = rand_strided((8, 1536, 7, 7), (75264, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_98 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_106 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_114 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_122 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_400 = rand_strided((8, 1536, 7, 7), (75264, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_130 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_138 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_146 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_154 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_162 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_469 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_170 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_178 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_186 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_194 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_528 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_202 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_210 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_218 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_226 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_587 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_234 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_242 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_646 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_266 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_274 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_282 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_290 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_705 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_306 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_314 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_764 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_338 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_354 = rand_strided((1, 384, 1), (384, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_362 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_833 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_370 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_378 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_386 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_892 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_410 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_418 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_426 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_434 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_961 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_450 = rand_strided((1, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_458 = rand_strided((1, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_466 = rand_strided((1, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_482 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_498 = rand_strided((1, 32, 1), (32, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_506 = rand_strided((1, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_174, primals_176, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, primals_204, primals_206, primals_208, primals_210, primals_212, primals_214, primals_216, primals_218, primals_222, squeeze_1, view_2, convolution, mul_3, squeeze_3, view_5, convolution_1, mul_7, squeeze_5, view_8, convolution_2, mul_11, squeeze_7, view_11, convolution_3, mul_16, squeeze_9, view_14, squeeze_11, view_17, convolution_5, mul_23, squeeze_13, view_20, convolution_6, mul_27, squeeze_15, view_23, convolution_7, mul_31, squeeze_17, view_26, convolution_8, mean, relu, convolution_10, mul_39, avg_pool2d, squeeze_19, view_29, squeeze_21, view_32, convolution_12, mul_46, squeeze_23, view_35, convolution_13, mul_50, squeeze_25, view_38, convolution_14, mul_54, squeeze_27, view_41, convolution_15, mean_1, relu_1, convolution_17, mul_62, squeeze_29, view_44, convolution_18, mul_66, squeeze_31, view_47, convolution_19, mul_70, squeeze_33, view_50, convolution_20, mul_74, squeeze_35, view_53, convolution_21, mean_2, relu_2, convolution_23, mul_82, avg_pool2d_1, squeeze_37, view_56, squeeze_39, view_59, convolution_25, mul_89, squeeze_41, view_62, convolution_26, mul_93, squeeze_43, view_65, convolution_27, mul_97, squeeze_45, view_68, convolution_28, mean_3, relu_3, convolution_30, mul_105, squeeze_47, view_71, convolution_31, mul_109, squeeze_49, view_74, convolution_32, mul_113, squeeze_51, view_77, convolution_33, mul_117, squeeze_53, view_80, convolution_34, mean_4, relu_4, convolution_36, mul_125, squeeze_55, view_83, convolution_37, mul_129, squeeze_57, view_86, convolution_38, mul_133, squeeze_59, view_89, convolution_39, mul_137, squeeze_61, view_92, convolution_40, mean_5, relu_5, convolution_42, mul_145, squeeze_63, view_95, convolution_43, mul_149, squeeze_65, view_98, convolution_44, mul_153, squeeze_67, view_101, convolution_45, mul_157, squeeze_69, view_104, convolution_46, mean_6, relu_6, convolution_48, mul_165, squeeze_71, view_107, convolution_49, mul_169, squeeze_73, view_110, convolution_50, mul_173, squeeze_75, view_113, convolution_51, mul_177, squeeze_77, view_116, convolution_52, mean_7, relu_7, convolution_54, mul_185, squeeze_79, view_119, convolution_55, mul_189, squeeze_81, view_122, convolution_56, mul_193, squeeze_83, view_125, convolution_57, mul_197, squeeze_85, view_128, convolution_58, mean_8, relu_8, convolution_60, mul_205, avg_pool2d_2, squeeze_87, view_131, squeeze_89, view_134, convolution_62, mul_212, squeeze_91, view_137, convolution_63, mul_216, squeeze_93, view_140, convolution_64, mul_220, squeeze_95, view_143, convolution_65, mean_9, relu_9, convolution_67, mul_228, squeeze_97, view_146, convolution_68, mul_232, squeeze_99, view_149, convolution_69, mul_236, squeeze_101, view_152, convolution_70, mul_240, squeeze_103, view_155, convolution_71, mean_10, relu_10, convolution_73, mul_248, squeeze_105, view_158, convolution_74, mul_252, squeeze_107, view_161, convolution_75, mul_256, squeeze_109, view_164, convolution_76, mul_260, squeeze_111, view_167, convolution_77, mean_11, relu_11, convolution_79, add_67, squeeze_113, view_170, convolution_80, clone_28, permute_1, unsqueeze_58, unsqueeze_66, unsqueeze_74, unsqueeze_82, unsqueeze_90, mul_341, unsqueeze_98, unsqueeze_106, unsqueeze_114, unsqueeze_122, mul_400, unsqueeze_130, unsqueeze_138, unsqueeze_146, unsqueeze_154, unsqueeze_162, mul_469, unsqueeze_170, unsqueeze_178, unsqueeze_186, unsqueeze_194, mul_528, unsqueeze_202, unsqueeze_210, unsqueeze_218, unsqueeze_226, mul_587, unsqueeze_234, unsqueeze_242, unsqueeze_250, unsqueeze_258, mul_646, unsqueeze_266, unsqueeze_274, unsqueeze_282, unsqueeze_290, mul_705, unsqueeze_298, unsqueeze_306, unsqueeze_314, unsqueeze_322, mul_764, unsqueeze_330, unsqueeze_338, unsqueeze_346, unsqueeze_354, unsqueeze_362, mul_833, unsqueeze_370, unsqueeze_378, unsqueeze_386, unsqueeze_394, mul_892, unsqueeze_402, unsqueeze_410, unsqueeze_418, unsqueeze_426, unsqueeze_434, mul_961, unsqueeze_442, unsqueeze_450, unsqueeze_458, unsqueeze_466, unsqueeze_474, unsqueeze_482, unsqueeze_490, unsqueeze_498, unsqueeze_506, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('nfnet_l0', benchmark_compiled_module)
