
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


# kernel path: /tmp/torchinductor_youkaichao/pf/cpfvamlk7ld2nnfqoekqagkildnga4ftul2tuckst7yetafacl3i.py
# Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_div_mul_native_batch_norm_backward_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_mul_native_batch_norm_backward_1', 'mutated_arg_names': []}
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
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1280*(r2 // 49)) + (2560*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (1280*r2) + (125440*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (1280*r2) + (125440*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 49.0
        tmp2 = tmp0 / tmp1
        tmp4 = tmp2 * tmp3
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


# kernel path: /tmp/torchinductor_youkaichao/gb/cgbmar44qm5t5l66sjy56o6oosdjwx3yh25cyz7ztkbvkmnubozz.py
# Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_div_mul_native_batch_norm_backward_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_mul_native_batch_norm_backward_2', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/t5/ct5r7x5hvxu3lok2kudb73zphbknx4poexi734lpbmpybl5hvyih.py
# Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_div_mul_native_batch_norm_backward_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_mul_native_batch_norm_backward_3', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/2u/c2u43nvyucqk3efmhgvdtge4vsgagzwhjmj5qj5cjinc5norwhfh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_div_mul_native_batch_norm_backward_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_mul_native_batch_norm_backward_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 1280
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y1 = (yindex // 49)
    y3 = yindex
    y0 = yindex % 49
    tmp0 = tl.load(in_ptr0 + (x2 + (1280*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (1280*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (1280*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 49.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 * tmp3
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
    tl.store(out_ptr0 + (y0 + (49*x2) + (62720*y1)), tmp21, xmask & ymask)
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


# kernel path: /tmp/torchinductor_youkaichao/h7/ch77boffzsrgqoc3lvjv6rqfxvxcceqalxlzo25ej3wolvkgmmwp.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate, x_267], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___6_____0___se_gate => sigmoid_63
# x_267 => mul_390, sigmoid_61
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1152
    x1 = (xindex // 1152)
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1152*r2) + (56448*x1)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = 1.0
    tmp12 = tmp11 - tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp8 * tmp13
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nc/cnc7ihrb4lmr6tosvwfls75vxwxt7qvlfryuifci2uihgqudvg7e.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1152*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fx/cfxluoqhyeidcqmkfu5nyeht4gxa72etql3kcz73b7h3rvvypaiq.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
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


# kernel path: /tmp/torchinductor_youkaichao/y4/cy4mg7gtpctfyd5bi5wknevxvfouugpxdgxsntius44agi7okoon.py
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
    size_hints=[64, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_12', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ay/cayem74bzaw5evnn5csmdymgnaqwujmvlmd54kjbutdrx52oauw2.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___6_____0___se_gate => sigmoid_63
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4608
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1152
    x1 = (xindex // 1152)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp20 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (56448*(r2 // 49)) + (112896*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (1152*(r2 // 49)) + (2304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (1152*(r2 // 49)) + (2304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (1152*r2) + (112896*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr4 + (x0 + (1152*r2) + (112896*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 49.0
        tmp6 = tmp4 / tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12 + tmp10
        tmp14 = tmp9 * tmp13
        tmp15 = tmp7 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp15 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dt/cdt5gym7po2kpzsoirv4ao6y7ajhrvzoyglv6ntksdl67wd4eo3f.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___6_____0___se_gate => sigmoid_63
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1152*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wl/cwl5lzdxl36r4fx3ozvlzdhfc7hb4c5jzimwny3hl5nrka6vmazh.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___6_____0___se_gate => sigmoid_63
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1152*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ah/cahvodvm3wdzhj33ou3libsfvjkrn4bdtrh2x3h3mugn4pvl5kwe.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___6_____0___se_gate => sigmoid_63
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 1152
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (56448*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (1152*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (1152*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (1152*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (1152*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp7 * tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.002551020408163265
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (1152*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ox/coxhgtmwwit52oq3ixumds24ef5j2gyuyhvcjhwnribccniz2uaw.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1152
    y1 = (yindex // 1152)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1152*x2) + (56448*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lx/clxfrscjbf5nk6lpprjstgkvt5vmuqjhkshd6yh4ss62cphpjn6v.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4608
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1152
    x1 = (xindex // 1152)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (56448*(r2 // 49)) + (112896*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (1152*r2) + (112896*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (1152*r2) + (112896*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
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


# kernel path: /tmp/torchinductor_youkaichao/eg/cegbddqfhlbfc2ssxd2kgperjlb3vqzut4leyu6vicge2aikfflh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1152
    y1 = (yindex // 1152)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (1152*x2) + (56448*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (1152*x2) + (56448*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
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
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jc/cjcd6aa7usoqizfarh4rqrkzh4cwe66ymoqzh56e45njwcf3wtzz.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (9408*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2o/c2o2egjkiuvdrnzbuziezdoe55l7kanverh5hsapt4kdyle2jy4f.py
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
    xnumel = 768
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (9408*(r2 // 49)) + (18816*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (192*r2) + (18816*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3a/c3aovlsvtlsttytijwjsbjrkdnvk4pzjlkg2bgtecw7f6z7ji5tl.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/6b/c6blkcqemb5bcx7aldldijxtwgqlc4t4w62afz35mxyhkknelwwf.py
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
    ynumel = 1536
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (192*x2) + (9408*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/43/c43gpmdh7agcc34wfivtis5q7kyetgxtgo5dp6e3ozzinpnharg3.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (9408*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (9408*r2)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0 + (192*r3)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/zp/czpe6svshv2x7jqnrju7rqvr7euvmohwteekls6uyhxrireehql2.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (192*x2) + (9408*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/wy/cwygz3gf44zrgjaly5ntimos7dxf5fagygezqbwqgie4sjozhvr7.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (9408*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (9408*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (49*x0) + (9408*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (x0 + (192*r3)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/v4/cv4rtwlx72n7wqi6c2e3cr2l4lv6o3j5zsfg4hz5eoquehqst27m.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (192*x2) + (9408*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/md/cmdceha7xxmuw5cajetpr5gxr5z27zqrqurj3jfyoiv5yinbjfi7.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (9408*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (9408*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (49*x0) + (9408*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (49*x0) + (9408*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr4 + (x0 + (192*r3)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/wr/cwr33ntxnoagp7wiq4m6osaak3ijrzsmvyz3nyn42q4dggw5wogd.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_29', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + (192*x2) + (9408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/2i/c2itp7crqrpwdxr5bmp55u3oewflx4rmby3iva3fvdq2ldjlrulc.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_200], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_47
# x_200 => mul_290, sigmoid_45
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_30', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 672
    x1 = (xindex // 672)
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (672*r2) + (32928*x1)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = 1.0
    tmp12 = tmp11 - tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp8 * tmp13
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cp/ccpgdz253cshdu5rpei7ij2tvbb6evmd7oqjymbzllawb7xq5aaf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7q/c7qmhb7tkjvukhkknawmlbyp4575rp2jo67hmppkp4r4huflwi74.py
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
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 224
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


# kernel path: /tmp/torchinductor_youkaichao/fu/cfugpvjsiuxewhawip37j2sl6vsk6kest42ajyxiwirrxvlu3cot.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 28
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (28*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dd/cddamwqevr3voga6rbkkqumajf25di6m6yevcgtwhwzdospaikni.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_47
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2688
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 672
    x1 = (xindex // 672)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp20 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (32928*(r2 // 49)) + (65856*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (672*(r2 // 49)) + (1344*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (672*(r2 // 49)) + (1344*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (672*r2) + (65856*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr4 + (x0 + (672*r2) + (65856*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 49.0
        tmp6 = tmp4 / tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12 + tmp10
        tmp14 = tmp9 * tmp13
        tmp15 = tmp7 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp15 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/no/cnojj6maf76zn5qcvfcmpsrtlapk73jubbcm5hmzyxorqdofhltr.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_47
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fg/cfgxlpajzy2dlrdee43r4nu2nxhdile7kvd4q5g65emcstsllasv.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_47
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j3/cj3opoiuhsboby66tgapec3wxdu4bzejmxmrlr567uuh2efaj7il.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_47
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 672
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (32928*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (672*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (672*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (672*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (672*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp7 * tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.002551020408163265
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (672*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u4/cu4ynouur5zovb3i7m5uutafvhdfds2edaro7uod5jpwcpucjxny.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 672
    y1 = (yindex // 672)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (672*x2) + (32928*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pi/cpiekhqql3g2sd352p2c4imflspqqzz4ykepx4p7abg6ngaz4y27.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8736
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = 1 + (((r2 + (121*x0)) // 14) % 14)
        tmp4 = tl.full([1, 1], 0, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tl.full([1, 1], 17, tl.int64)
        tmp7 = tmp3 < tmp6
        tmp8 = 1 + ((r2 + (121*x0)) % 14)
        tmp9 = tmp8 >= tmp4
        tmp10 = tmp8 < tmp6
        tmp11 = tmp5 & tmp7
        tmp12 = tmp11 & tmp9
        tmp13 = tmp12 & tmp10
        tmp14 = tmp13 & tmp2
        tmp15 = tl.load(in_ptr0 + (18 + (17*(((r2 + (121*x0)) // 14) % 14)) + (289*x1) + (194208*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 14)), rmask & tmp14 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
        tmp17 = tl.where(tmp14, tmp15, tmp16)
        tmp18 = tl.load(in_ptr1 + (x1 + (672*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tmp17 * tmp18
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ks/cksmyuffn4h3jyvrnkvyoeouluac2cdwfdiybjlsuwgnrcaw4tcq.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
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


# kernel path: /tmp/torchinductor_youkaichao/zx/czxyusu3o2e4asv5ihawt6tshtg6emeebpclv7bmaivyq4ka6kmd.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8736
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 672)
    x0 = xindex % 672
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = 1 + (((r2 + (121*x1)) // 14) % 14)
        tmp4 = tl.full([1, 1], 0, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tl.full([1, 1], 17, tl.int64)
        tmp7 = tmp3 < tmp6
        tmp8 = 1 + ((r2 + (121*x1)) % 14)
        tmp9 = tmp8 >= tmp4
        tmp10 = tmp8 < tmp6
        tmp11 = tmp5 & tmp7
        tmp12 = tmp11 & tmp9
        tmp13 = tmp12 & tmp10
        tmp14 = tmp13 & tmp2
        tmp15 = tl.load(in_ptr0 + (18 + (17*(((r2 + (121*x1)) // 14) % 14)) + (289*x0) + (194208*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 14)), rmask & tmp14 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
        tmp17 = tl.where(tmp14, tmp15, tmp16)
        tmp18 = tl.load(in_ptr1 + (x0 + (672*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tmp17 * tmp18
        tmp20 = tl.load(in_ptr2 + (x0 + (672*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tmp20 - tmp21
        tmp23 = tmp19 * tmp22
        tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
        tmp25 = tl.where(tmp2, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sh/cshyiolagh6lga65osdnulw7n3bbm5z3as5c3m4e2fmxbdyx4l4i.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yy/cyyea5dqjnkfbowlb53kksktrwxcnpvniia5vnqbmct7xko4alhb.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 14)
    x2 = xindex % 14
    y4 = yindex
    x5 = xindex
    y0 = yindex % 672
    y1 = (yindex // 672)
    tmp14 = tl.load(in_ptr1 + (y0 + (672*x5) + (131712*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (y0 + (672*x5) + (131712*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = 1 + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 17, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (18 + x2 + (17*x3) + (289*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp15 = tmp13 * tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.0006377551020408163
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.store(out_ptr0 + (x5 + (196*y4)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fv/cfv3gq6tgg4gnoub5jp25ibbyraz4n4yo2odb64eq5wbm4pg66tc.py
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
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 112
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (21952*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bc/cbcwvszzuqaxqmjc7uev3hr4hheogt64nlpoovhmhxpb6ykrw4oe.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1456
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
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (21952*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (112*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/7o/c7ogwibucd7sswf5mdave3iacrgiusb3j76zi44wdkpha7ozadmr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 112
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


# kernel path: /tmp/torchinductor_youkaichao/jk/cjkbodd762shwcp5e2kek2zrghvvu44cn7hob2gui3rgspvehe7w.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (112*x2) + (21952*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/3r/c3rk5dn3kxb2imluo5vlt6hinc5vueepntgquvly74smfdfddj4i.py
# Source Nodes: [x_181], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_181 => mul_265, sigmoid_41
triton_red_fused_mul_silu_sum_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10752
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 2
    x1 = (xindex // 2) % 672
    x2 = (xindex // 1344)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (98*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (672*r3) + (65856*x0) + (131712*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m7/cm7ea67ayqvxvh74zh4lvnnsgmfj6qdgkhyjasbmep7fe4v6wcyn.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_181], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___4_____2___se_gate => sigmoid_43
# x_181 => mul_265, sigmoid_41
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_49', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tmp6 * tmp8
    tmp10 = tmp4 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5y/c5y4gcxignxzzydu5rmiibrwjkra3uizsmcyc7fsxhfqop5ihst6.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____2___se_gate => sigmoid_43
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8736
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (131712*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (672*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.load(in_ptr2 + (x1 + (672*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = 196.0
        tmp9 = tmp7 / tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tl.load(in_ptr3 + (x1 + (672*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.sigmoid(tmp11)
        tmp13 = 1.0
        tmp14 = tmp13 - tmp12
        tmp15 = tmp11 * tmp14
        tmp16 = tmp15 + tmp13
        tmp17 = tmp12 * tmp16
        tmp18 = tmp10 * tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7a/c7aoqj53q5enxwwzkcqkxd5j4excf3gvyghdgyoxqyhi62ihause.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____2___se_gate => sigmoid_43
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8736
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 672)
    x0 = xindex % 672
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x0) + (131712*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (672*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.load(in_ptr2 + (x0 + (672*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = 196.0
        tmp9 = tmp7 / tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tl.load(in_ptr3 + (x0 + (672*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.sigmoid(tmp11)
        tmp13 = 1.0
        tmp14 = tmp13 - tmp12
        tmp15 = tmp11 * tmp14
        tmp16 = tmp15 + tmp13
        tmp17 = tmp12 * tmp16
        tmp18 = tmp10 * tmp17
        tmp19 = tl.load(in_ptr4 + (x0 + (672*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp18 * tmp21
        tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
        tmp24 = tl.where(tmp2, tmp22, tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yw/cywmgouwvn23yfgu6r7wl6ci5skffhkhzmykkiwqtaxibfzdxlfd.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____2___se_gate => sigmoid_43
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 672
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (131712*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (672*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (672*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (672*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (672*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp7 * tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.0006377551020408163
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (672*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ta/ctawheweyjycsuojeka6ajopssmc5ckfv4xcvn5bqgdn36wfeash.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 672
    y1 = (yindex // 672)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (672*x2) + (131712*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lt/clttipscg2vpvdrw4rv3f6dv33zmlwgsqic4dfoyh62755k6tb7f.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8736
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (131712*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (672*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qf/cqfhlc56dg7gs3x2y2k33sgqaf4dzbqworuztae2fsarjj74rw5n.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8736
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 672)
    x0 = xindex % 672
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x0) + (131712*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (672*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.load(in_ptr2 + (x0 + (672*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7llbqivpvoybfwhs6w5slqqducinttyoyaj3vulnrxvacutwnv2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_56', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 672
    y1 = (yindex // 672)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (672*x2) + (131712*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (672*x2) + (131712*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
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


# kernel path: /tmp/torchinductor_youkaichao/rv/crvh7xxx7kddk5j23wa5xuwqlv7cbk5qdjcg5ohxvap4grjwskbi.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 112
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (21952*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (21952*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (112*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/iy/ciyhoxhccghaevnboxjfoagmcntyse2f6cqsqnd6w6ti3hcnlkyb.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (112*x2) + (21952*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/p7/cp7ju4x2jqp7vn5pf2gggfgtwwvvaydphrhvhffco7pmhhcetnza.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 112
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (21952*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (21952*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (196*x0) + (21952*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (112*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/vq/cvqqtpu3dtaq55dvj6kgdd63evgmdab65cftlzltx536xvrnn3wb.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_60', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + (112*x2) + (21952*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/mc/cmcp5mnax6uhmv2b6pvmzq6ylwgdd22mqszreglem7n5vuaira4o.py
# Source Nodes: [x_148], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_148 => mul_215, sigmoid_33
triton_red_fused_mul_silu_sum_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 2
    x1 = (xindex // 2) % 480
    x2 = (xindex // 960)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (98*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (480*r3) + (47040*x0) + (94080*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mo/cmohfvdd2fgndfdxxsjeythnp36dskxmarjf7pf3g77m2vumbemz.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_148], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
# x_148 => mul_215, sigmoid_33
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_62 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_62', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tmp6 * tmp8
    tmp10 = tmp4 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ln/cln5vegx7aoyhg2wql63qvgetigazcqcjgvtsqe4eztliiqfrxw2.py
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
    size_hints=[512, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (480*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z3/cz3ekv44hk32bqluth2fmueny62t37ibze47elh4qbdbslhlznqf.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_64', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
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


# kernel path: /tmp/torchinductor_youkaichao/cv/ccvqb4nak6eoyzqzuej5cyfpsi4ofylreitqu3z2rgql6g2pic6c.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 20
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (20*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ea/ceaufo2bgpnqcljtsjjliotg2gr2jo5dk5vbhuc3d5xo3bbfn2ho.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6240
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (94080*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (480*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.load(in_ptr2 + (x1 + (480*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = 196.0
        tmp9 = tmp7 / tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tl.load(in_ptr3 + (x1 + (480*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.sigmoid(tmp11)
        tmp13 = 1.0
        tmp14 = tmp13 - tmp12
        tmp15 = tmp11 * tmp14
        tmp16 = tmp15 + tmp13
        tmp17 = tmp12 * tmp16
        tmp18 = tmp10 * tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ir/cirpj7unbgdtukcdu6hzikpbwk5x6eiuihlpv3txfwsz4xxlllhs.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
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


# kernel path: /tmp/torchinductor_youkaichao/va/cvam2ctgl6cuuwoqzjonvqgtxiacxrmu5qvtp5putydywghldm5q.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6240
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 480)
    x0 = xindex % 480
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x0) + (94080*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (480*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.load(in_ptr2 + (x0 + (480*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = 196.0
        tmp9 = tmp7 / tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tl.load(in_ptr3 + (x0 + (480*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.sigmoid(tmp11)
        tmp13 = 1.0
        tmp14 = tmp13 - tmp12
        tmp15 = tmp11 * tmp14
        tmp16 = tmp15 + tmp13
        tmp17 = tmp12 * tmp16
        tmp18 = tmp10 * tmp17
        tmp19 = tl.load(in_ptr4 + (x0 + (480*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp18 * tmp21
        tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
        tmp24 = tl.where(tmp2, tmp22, tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ts/ctse77bspmhblopdirwazbegxww5v4rklwalq6uutea5embjd2r5.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (480*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ub/cubyszbdhhoey2bx6e5zfcw7kby3i5eaphygbocfrec4pl5laxxx.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 480
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (480*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (480*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (480*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (480*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp7 * tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.0006377551020408163
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (480*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ih/cihtker2filng5a2a27nlgco2ciza5ak2l7xlwz7zfxxe3wqcosp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 480
    y1 = (yindex // 480)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (480*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/du/cdunyr5a53yaydyqq2ixb6x2hhrf4phx3kqoq47uz3x3bg5xzlal.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6240
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (94080*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (480*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iq/ciqsamuqaoiwugox75yw5l4di2skwrvgzpir5errq5dnsmafcjtg.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6240
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 480)
    x0 = xindex % 480
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x0) + (94080*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (480*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.load(in_ptr2 + (x0 + (480*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qt/cqtke3zn3xctirs3eachdh5xyhpbhoevzy5rkvmkjicdkqwwqdro.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_74 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_74', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 480
    y1 = (yindex // 480)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (480*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (480*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
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


# kernel path: /tmp/torchinductor_youkaichao/ol/coltcsttfvvukdssbqlhyco5pfr6lacvhzqh277njnvda4e2u4ys.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ha/cha3tqub7p6ftxu6veghha5kcew66vv2rjui7yffqlm6w6mb36qn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1040
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
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (15680*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (80*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/uk/cuk7mpvubvlzqmcrrjdvww2qzuqnlca3a2hkftmt3ouvsxarpcsb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 80
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


# kernel path: /tmp/torchinductor_youkaichao/ph/cphgto5sf337b345lxxat3wiv4ii4lbjeaacyshnkxa2qjspundq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 80
    y1 = (yindex // 80)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (80*x2) + (15680*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/zh/czhksbnlngpjx7zzq67af7ykgkare3odbckncgidnxpuxfnsncsf.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_79', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (80*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/2l/c2lt74klilofmlwavihj4obvx2qochgddpafa2dtrlqxa2gmvhtn.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 80
    y1 = (yindex // 80)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (80*x2) + (15680*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/r3/cr3pv67lgwb4cjrjqv7n2psp7lcr3kpk6dyfgb5chxpeyn3lpcta.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (80*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/u5/cu5iatdmzpzsw5velxgnxo2h5tezcamhrbsdso3o5lmqipdjozc4.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_82', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 80
    y1 = (yindex // 80)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + (80*x2) + (15680*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/ti/ctiotfvtmkpfbyb6sijadfenj4ea4sbpzoc2v6ctiymo6pglnneb.py
# Source Nodes: [x_98], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_98 => mul_140, sigmoid_21
triton_red_fused_mul_silu_sum_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_83', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 2
    x1 = (xindex // 2) % 240
    x2 = (xindex // 480)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (98*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (240*r3) + (23520*x0) + (47040*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2r/c2rztjiuzyi6lcwkmm6vwqpvooeucxrayixufpsjbdrvedqefid3.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_98], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_23
# x_98 => mul_140, sigmoid_21
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_84 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_84', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tmp6 * tmp8
    tmp10 = tmp4 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bh/cbhagmqqfjporpnsanvay27osgtvlhl26n4snyjxd5366rqsepcs.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (240*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bk/cbkjntgvd33u4eyto7b7u63lkjetg4kboy4wtsc6ulohbguvxjb4.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_86', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 80
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


# kernel path: /tmp/torchinductor_youkaichao/j6/cj6r3smvis47ad4g234sfrb7vczma4zidezdevfvtsqauk56lovj.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 10
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (10*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5h/c5hcn4gmkbazwnocblv35fuhl3drxavudzzgelsxresmokx7jdm6.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_23
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_88 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_88', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3120
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (47040*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (240*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.load(in_ptr2 + (x1 + (240*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = 196.0
        tmp9 = tmp7 / tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tl.load(in_ptr3 + (x1 + (240*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.sigmoid(tmp11)
        tmp13 = 1.0
        tmp14 = tmp13 - tmp12
        tmp15 = tmp11 * tmp14
        tmp16 = tmp15 + tmp13
        tmp17 = tmp12 * tmp16
        tmp18 = tmp10 * tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tc/ctc6idzh5akc6ucffnjdr45x22rzo4k6gwyc5hzyzwf2smacp6sk.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_23
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
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


# kernel path: /tmp/torchinductor_youkaichao/cq/ccqd4ksdh3wu4agbvvsiee2v6fwnutk5rzutrhiyo6nlnvqc6cij.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_23
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_90 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_90', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3120
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 240)
    x0 = xindex % 240
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x0) + (47040*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (240*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.load(in_ptr2 + (x0 + (240*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = 196.0
        tmp9 = tmp7 / tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tl.load(in_ptr3 + (x0 + (240*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.sigmoid(tmp11)
        tmp13 = 1.0
        tmp14 = tmp13 - tmp12
        tmp15 = tmp11 * tmp14
        tmp16 = tmp15 + tmp13
        tmp17 = tmp12 * tmp16
        tmp18 = tmp10 * tmp17
        tmp19 = tl.load(in_ptr4 + (x0 + (240*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp18 * tmp21
        tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
        tmp24 = tl.where(tmp2, tmp22, tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yu/cyudntxex2woh3ruclhnqy7ffurlwmlwgrq7qcljhkdla2tu735k.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_23
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (240*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2i/c2i2rko7sba4y5lxct6ces75yc33mk6edmn3to252top4ba2lzea.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_23
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_92 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_92', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 240
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (47040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (240*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (240*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (240*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (240*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp7 * tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.0006377551020408163
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (240*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mm/cmmhdpedxkqbpkmgm6t4fx7swtpnlkjuvbttsvdhyduehmbbttcc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_93 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_93', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (240*x2) + (47040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/34/c34uetzu7kmg6iiax5thn2taifduxhug2ffdyfxwlsu5dd6nwesp.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_94 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_94', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 11760
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr1 + (x1 + (240*r2) + (30720*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = ((r2 + (128*x0)) // 28) % 28
        tmp1 = tl.full([1, 1], 29, tl.int64)
        tmp2 = tmp0 < tmp1
        tmp3 = (r2 + (128*x0)) % 28
        tmp4 = tmp3 < tmp1
        tmp5 = tmp2 & tmp4
        tmp6 = tl.load(in_ptr0 + ((29*(((r2 + (128*x0)) // 28) % 28)) + (841*x1) + (201840*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 28)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp10 = tmp8 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mx/cmx6l3n6htfggkxwfrfmqqw3qeqt3mezi26kuzeizcm5dppmbur4.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_95', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
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


# kernel path: /tmp/torchinductor_youkaichao/ip/cipke3v3qo4rsoqciunngkigsumwkcj6bteymc6np7ddvr43cbb5.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_96 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_96', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 11760
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 240)
    x0 = xindex % 240
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr1 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr2 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = ((r2 + (128*x1)) // 28) % 28
        tmp1 = tl.full([1, 1], 29, tl.int64)
        tmp2 = tmp0 < tmp1
        tmp3 = (r2 + (128*x1)) % 28
        tmp4 = tmp3 < tmp1
        tmp5 = tmp2 & tmp4
        tmp6 = tl.load(in_ptr0 + ((29*(((r2 + (128*x1)) // 28) % 28)) + (841*x0) + (201840*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 28)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp10 = tmp8 * tmp9
        tmp13 = tmp11 - tmp12
        tmp14 = tmp10 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d6/cd63f3ars6n2z3i5bvq7ir3atmfh6d6fqvbik43utx2rdj5c6iz4.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_97', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (240*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/up/cupf3qjozchtzxhde4y6eexziz6suqvbpqadhsxontfqktucm2iv.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_98', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 28)
    x2 = xindex % 28
    y4 = yindex
    x5 = xindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp9 = tl.load(in_ptr1 + (y0 + (240*x5) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0 + (240*x5) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = x3
    tmp1 = tl.full([1, 1], 29, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x2
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + (29*x3) + (841*y4)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp10 = tmp8 * tmp9
    tmp13 = tmp11 - tmp12
    tmp15 = 0.00015943877551020407
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(out_ptr0 + (x5 + (784*y4)), tmp27, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7o/c7o72rxx6axkg7tf23c2kmovagk37qzca56gz4nymqlefrysmebl.py
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
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_99', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/rq/crqg3wr4hccefyuycx4hrko3v6diuolkwb6lmspjpy4mp2cxhbca.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_100 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_100', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/uw/cuwlxlcte47sv6t54qeiv73hu2icazb6g6sswvmfgjqatdpbqomk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_101 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_101', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ft/cfty5fpujuctkghk6h75qvjq7fc4vhzqyfbwjnfued5gd2qwiqwr.py
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
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_102', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/to/cto4qrktl54qrfkqeayshzifygxe2fdgilyqxtacqtt5h5g5devw.py
# Source Nodes: [x_79], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_79 => mul_115, sigmoid_17
triton_red_fused_mul_silu_sum_103 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_103', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 13440
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 7
    x1 = (xindex // 7) % 240
    x2 = (xindex // 1680)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (112*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (240*r3) + (26880*x0) + (188160*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/24/c24e4puusblhpdzt2w6qqp57gz2wrsb7lv3hceh4p7sydhbganhl.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_79], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___2_____1___se_gate => sigmoid_19
# x_79 => mul_115, sigmoid_17
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_104 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_104', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
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
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tmp6 * tmp8
    tmp10 = tmp4 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yw/cywqq3pxivc4w3652pbf3uxxaojlukz3uj2zy4aingz7zym6v776.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____1___se_gate => sigmoid_19
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_105 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_105', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 11760
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (188160*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (240*((r2 + (128*x0)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (240*((r2 + (128*x0)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (240*r2) + (30720*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 784.0
        tmp6 = tmp4 / tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12 + tmp10
        tmp14 = tmp9 * tmp13
        tmp15 = tmp7 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qe/cqewqer4kyhpzc5dnk3b7oaujlex4m5bqoj7dsystjataptceqxk.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____1___se_gate => sigmoid_19
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_106 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_106', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 11760
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 240
    x1 = (xindex // 240)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x0) + (188160*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (240*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (240*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 784.0
        tmp6 = tmp4 / tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12 + tmp10
        tmp14 = tmp9 * tmp13
        tmp15 = tmp7 * tmp14
        tmp18 = tmp16 - tmp17
        tmp19 = tmp15 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gt/cgtxrepzntvclxypmdgmx34fkhe4jm5xjedhtix357ukb3k34ye5.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____1___se_gate => sigmoid_19
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_107 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_107', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 240
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (240*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (240*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (240*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (240*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp7 * tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.00015943877551020407
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (240*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lw/clw6yehgrsxxmrfoducuzk6rwmz6h6rlcozmon4twmkwfwsfu7w5.py
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
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_108', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (240*x2) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ft/cftzxq2254zbs3sq73n5ljdwxqeflyovuc3ijepqicnzlpjvwuas.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_109 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_109', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 11760
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (188160*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (240*r2) + (30720*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t4/ct42j75nrrdq6wcqvgzf6agcfaajimq57hg7ftdxbwqmpwjpq36i.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_110', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 11760
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 240
    x1 = (xindex // 240)
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x0) + (188160*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6t/c6tmekkyaepd5xbvh7rdbob3u6gzvr5jhtbvhybaddl5yamqyhno.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_111 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_111', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (240*x2) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (240*x2) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6l/c6lugbjxzrgfa5kebpnik7pgmcpnftkinqn5fhfqrhqouaohfvtz.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_112 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_112', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/dv/cdvcwmnledhtw6j7optyu3xi6wjxbfj5x52bix6ddtdyaeobvo45.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_113 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_113', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (40*x2) + (31360*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xn/cxnzb5tquiinytqb2dlx3o5sl4egfnmazr3g56vspbzw54twy4o6.py
# Source Nodes: [x_63], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_63 => mul_90, sigmoid_13
triton_red_fused_mul_silu_sum_114 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_114', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8064
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 7
    x1 = (xindex // 7) % 144
    x2 = (xindex // 1008)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (112*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (144*r3) + (16128*x0) + (112896*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b2/cb2nqlkd4relvy4ktw66mwfkhno2slq2a2t5kcan3s5grwpxw773.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_63], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_15
# x_63 => mul_90, sigmoid_13
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_115 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_115', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
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
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tmp6 * tmp8
    tmp10 = tmp4 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gt/cgtw4h3e6ppzx4yehfccplaxq34ugpqaxn5tvmlbfkq7fwtnz6nv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_116 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_116', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (144*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ks/cks6x2xduv6wgvemfws2nbiuezz4i2nylo67j3uuskbx5sctkqu3.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_117 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_117', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48
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


# kernel path: /tmp/torchinductor_youkaichao/yd/cydtcbf7mp6eghotu6ra2qsglfyhjmfjfjqrodmfzclek6iqv7rk.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_118 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_118', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (6*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wt/cwtyt3wsdyfrd2emmznwh3shcryx56iyw3m4xsxbcidfuntjiwek.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_15
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_119 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_119', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7056
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (112896*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (144*((r2 + (128*x0)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (144*((r2 + (128*x0)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (144*r2) + (18432*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 784.0
        tmp6 = tmp4 / tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12 + tmp10
        tmp14 = tmp9 * tmp13
        tmp15 = tmp7 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a3/ca3q2tmsj5idmgf647b5zgmx4em7wgcfzsldhk3qi2mjoufnkk3p.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_15
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_120 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_120', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/o2/co2vurfkiompzqkbl4qoxqokinunoj7btenvck3i4nidbzqzrwyu.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_15
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_121 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_121', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7056
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 144
    x1 = (xindex // 144)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x0) + (112896*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (144*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (144*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 784.0
        tmp6 = tmp4 / tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12 + tmp10
        tmp14 = tmp9 * tmp13
        tmp15 = tmp7 * tmp14
        tmp18 = tmp16 - tmp17
        tmp19 = tmp15 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5d/c5dcy3ffhbcsg7zn6mkru7g2obwtf7r66ankjvidnehfsfdsg6bt.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_15
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_122 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_122', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ie/ciep3mjxqzb3njyzajupz4c4vltsu56llfbxpqgz5rpsfpgijkoo.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_15
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_123 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_123', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 144
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (112896*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (144*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (144*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp7 * tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.00015943877551020407
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (144*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ou/couzkusnpppkgppgpizayyrrf2phdhdulm6j3a4yfzvamnxrcgny.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_124 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_124', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 144
    y1 = (yindex // 144)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (144*x2) + (112896*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zd/czd4v3ozzmonmcvl5ys3nchipp62rlwfyjj7x2qrim44vgqz5ein.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_125 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_125', 'mutated_arg_names': []}
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
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp14 = tl.load(in_ptr1 + (x1 + (144*r2) + (18432*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = 1 + (((r2 + (128*x0)) // 56) % 56)
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 59, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = 1 + ((r2 + (128*x0)) % 56)
        tmp6 = tmp5 >= tmp1
        tmp7 = tmp5 < tmp3
        tmp8 = tmp2 & tmp4
        tmp9 = tmp8 & tmp6
        tmp10 = tmp9 & tmp7
        tmp11 = tl.load(in_ptr0 + (60 + (59*(((r2 + (128*x0)) // 56) % 56)) + (3481*x1) + (501264*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 56)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp10, tmp11, tmp12)
        tmp15 = tmp13 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ii/cii6tt2yay3qbccu3w4zqiu2qi3262x33f3ghatw4s2oytvfoyqd.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_126 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_126', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/r4/cr4iblaawg46ygv5xedpbomsee4u333re7cj6lgnb5xhykenxhjd.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_127 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_127', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 28224
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 144)
    x0 = xindex % 144
    tmp17 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp14 = tl.load(in_ptr1 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr2 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = 1 + (((r2 + (128*x1)) // 56) % 56)
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 59, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = 1 + ((r2 + (128*x1)) % 56)
        tmp6 = tmp5 >= tmp1
        tmp7 = tmp5 < tmp3
        tmp8 = tmp2 & tmp4
        tmp9 = tmp8 & tmp6
        tmp10 = tmp9 & tmp7
        tmp11 = tl.load(in_ptr0 + (60 + (59*(((r2 + (128*x1)) // 56) % 56)) + (3481*x0) + (501264*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 56)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp10, tmp11, tmp12)
        tmp15 = tmp13 * tmp14
        tmp18 = tmp16 - tmp17
        tmp19 = tmp15 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sj/csj2fizwnit22llytpjszs5agbmgyp7c7hrxe23lv2tlolxmfcrr.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_128 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_128', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/gc/cgcottn6yrucxvazcopdq47tap7knghkkkikaizhrnz6ogycbhkm.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_129 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_129', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 56)
    x2 = xindex % 56
    y4 = yindex
    x5 = xindex
    y0 = yindex % 144
    y1 = (yindex // 144)
    tmp14 = tl.load(in_ptr1 + (y0 + (144*x5) + (451584*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (y0 + (144*x5) + (451584*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = 1 + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 59, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (60 + x2 + (59*x3) + (3481*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp15 = tmp13 * tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 3.985969387755102e-05
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.store(out_ptr0 + (x5 + (3136*y4)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p2/cp24rkfstzx7ojcgvmnqpcyq5v3p3vaipa3mjeuixh4spf6n2jv6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_130 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_130', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/vf/cvfvzlxnmfgshd4t6b5co3e733le6d5qlydl4i7vqvfhjmt5s5x5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_131 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_131', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ic/cic6ke72exuxovyv6vslwqiftsbp73g35d4odc2mwlapxvumiizm.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_132 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_132', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ts/ctskipdysqcjjaidhy57fxvpwilst4uvwgm5ksgktpht4ic3iqwk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_133 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_133', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/pm/cpm52agllo6hayub5vlfqskzwyba4uiv5u4exddhghog5cndy5pp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_134 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_134', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/kk/ckkggrutnyupzaqib2c4gshtbgqcbo2r35fggrtjpfoebagl6eav.py
# Source Nodes: [x_44], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_44 => mul_65, sigmoid_9
triton_red_fused_mul_silu_sum_135 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_135', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 28800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 25
    x4 = (xindex // 25)
    x1 = (xindex // 25) % 144
    x2 = (xindex // 3600)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (126*x0)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((3136*x4) + ((r3 + (126*x0)) % 3136)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (144*((r3 + (126*x0)) % 3136)) + (451584*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp4 * tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3y/c3ynwa3kobzqyfiniy3sv3nbpe5qtli4jz75v37gtbmggsi5ybe7.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_44], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___1_____1___se_gate => sigmoid_11
# x_44 => mul_65, sigmoid_9
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_136 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_136', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
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
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tmp6 * tmp8
    tmp10 = tmp4 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qb/cqb7n44p2myyx4nsnoeyrg5bfrcvj7xbgk6hzffq72st4mz6pjek.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___1_____1___se_gate => sigmoid_11
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_137 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_137', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 28224
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x1) + (451584*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (144*((r2 + (128*x0)) // 3136))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (144*((r2 + (128*x0)) // 3136))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (144*r2) + (18432*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 3136.0
        tmp6 = tmp4 / tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12 + tmp10
        tmp14 = tmp9 * tmp13
        tmp15 = tmp7 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ig/ciggptwhq4rxigkghvlh4dx477vwuoy6xbqpdj55r6ajmplo52nz.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___1_____1___se_gate => sigmoid_11
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_138 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_138', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 28224
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 144
    x1 = (xindex // 144)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (451584*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (144*((r2 + (128*x1)) // 3136))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (144*((r2 + (128*x1)) // 3136))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 3136.0
        tmp6 = tmp4 / tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12 + tmp10
        tmp14 = tmp9 * tmp13
        tmp15 = tmp7 * tmp14
        tmp18 = tmp16 - tmp17
        tmp19 = tmp15 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/de/cdee5yxvot76jr7x656x4rj6foc2fzsz6izypl2k2fcc6hd7i3dc.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___1_____1___se_gate => sigmoid_11
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_139 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_139', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 144
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
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (451584*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (144*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (144*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 3136.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp7 * tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 3.985969387755102e-05
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (144*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pc/cpcssyz5tyvl6ujnfghwxx6cdqjmtpvhyh2kdiue2richmsqo3ff.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_140 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_140', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 144
    y1 = (yindex // 144)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (144*x2) + (451584*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/53/c53c4hnoh5pojx2qtw4fyyjljtkglolpwpf5egwu7jhoq4hutd6z.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_141 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_141', 'mutated_arg_names': []}
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
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x1) + (451584*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (144*r2) + (18432*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i3/ci35jvwgsiutjhnw62rtg2bacnvsfa4gajko7dz2s5ugujke2kmy.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_142 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_142', 'mutated_arg_names': []}
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
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (451584*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7y/c7yqn7ggsutvjq7jr4rdm5wwzq5nrg4xkjsak6bek35yozyalfg5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_143 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_143', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 144
    y1 = (yindex // 144)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (144*x2) + (451584*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (144*x2) + (451584*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
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


# kernel path: /tmp/torchinductor_youkaichao/7t/c7tbk76eslq345upo4phe7ehpgsavajrthugxubgzuecz62llyuy.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_144 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_144', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/dd/cddanuhof2e545g5eygd43q6tvumhcfzgekkfuzmgm5cpf6ipsn5.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_145 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_145', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/dt/cdtzpttl77x66cwgjvp3vyt5sojlbvamx6aeuvormaeofppfhkxm.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_146 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_146', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/rz/crztln6oldjfccqjdw5on3oqpchqx45f7zreefavyoutuait4uqp.py
# Source Nodes: [x_28], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_28 => mul_40, sigmoid_5
triton_red_fused_mul_silu_sum_147 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_147', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 19200
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 25
    x4 = (xindex // 25)
    x1 = (xindex // 25) % 96
    x2 = (xindex // 2400)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (126*x0)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((3136*x4) + ((r3 + (126*x0)) % 3136)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (96*((r3 + (126*x0)) % 3136)) + (301056*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp4 * tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/us/cus4bssdiafsijrxbicqwg5rv6wipbtsffciplic3acuko5xbboq.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_28], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___1_____0___se_gate => sigmoid_7
# x_28 => mul_40, sigmoid_5
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_148 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_148', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
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
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tmp6 * tmp8
    tmp10 = tmp4 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dr/cdruihevhhe2ldenmx5gie5pjrzhlsly5b7flre2furxkmy5gnvp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_149 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_149', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ao/cao2sk2e6ufeuzs5k24667mxvdq533wqt77vlenrjrw4zxv3sspn.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_150 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_150', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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


# kernel path: /tmp/torchinductor_youkaichao/cz/ccziedm5axqdktk2rw4llxxyphhqu27njuwg4236vmqz5x474i6a.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_151 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_151', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (4*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5k/c5ksseiwvm4gilibpv74kwjonzgnlsm7mjepdf5urgcfredclwa3.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___1_____0___se_gate => sigmoid_7
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_152 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_152', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18816
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x1) + (301056*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (96*((r2 + (128*x0)) // 3136))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (96*((r2 + (128*x0)) // 3136))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (96*r2) + (12288*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 3136.0
        tmp6 = tmp4 / tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12 + tmp10
        tmp14 = tmp9 * tmp13
        tmp15 = tmp7 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bl/cblymqwcctzqvkz2bj7rr7hcsbfvf6j3k2pdbfocqcdm5k5li76x.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___1_____0___se_gate => sigmoid_7
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_153 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_153', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/qq/cqqp7r7xxc4ha6otsykfm4sityx5g5ju4g4rx3puntk25zfcwdkh.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___1_____0___se_gate => sigmoid_7
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_154 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_154', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18816
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 96
    x1 = (xindex // 96)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (301056*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (96*((r2 + (128*x1)) // 3136))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (96*((r2 + (128*x1)) // 3136))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 3136.0
        tmp6 = tmp4 / tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12 + tmp10
        tmp14 = tmp9 * tmp13
        tmp15 = tmp7 * tmp14
        tmp18 = tmp16 - tmp17
        tmp19 = tmp15 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gv/cgv3tqcj5ugqfquhxxtlrvtttbkhwmjhomlzyrbs7vdcxifgajqa.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___1_____0___se_gate => sigmoid_7
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_155 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_155', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ei/ceiqnmhe3vg6ronzqqshnv6l6xlgua7kmcktl4v2efvaqgilyh2t.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___1_____0___se_gate => sigmoid_7
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_156 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_156', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 96
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
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (96*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (96*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 3136.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp7 * tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 3.985969387755102e-05
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (96*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k4/ck46wm4se3hasyyvmbwzq6ysdsxocfabshch2ngzpcbajatzf3ye.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_157 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_157', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (96*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z5/cz5ozydk67vssvdjf3qrisvmyxndgfhukurpmfr76mvpn5bww37r.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_158 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_158', 'mutated_arg_names': []}
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
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr1 + (x1 + (96*r2) + (12288*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = ((r2 + (128*x0)) // 112) % 112
        tmp1 = tl.full([1, 1], 113, tl.int64)
        tmp2 = tmp0 < tmp1
        tmp3 = (r2 + (128*x0)) % 112
        tmp4 = tmp3 < tmp1
        tmp5 = tmp2 & tmp4
        tmp6 = tl.load(in_ptr0 + ((113*(((r2 + (128*x0)) // 112) % 112)) + (12769*x1) + (1225824*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 112)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp10 = tmp8 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7r/c7rlzzv4iplhy6yd3dgcs4moizpgrqvs4je26pxw74dztv33w4kd.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_159 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_159', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/si/csip4mllaeskn567hspb7usfny3xpoedrw5ehpr6gxinfvl6ln4e.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_160 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_160', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 75264
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96)
    x0 = xindex % 96
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr1 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr2 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = ((r2 + (128*x1)) // 112) % 112
        tmp1 = tl.full([1, 1], 113, tl.int64)
        tmp2 = tmp0 < tmp1
        tmp3 = (r2 + (128*x1)) % 112
        tmp4 = tmp3 < tmp1
        tmp5 = tmp2 & tmp4
        tmp6 = tl.load(in_ptr0 + ((113*(((r2 + (128*x1)) // 112) % 112)) + (12769*x0) + (1225824*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 112)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp10 = tmp8 * tmp9
        tmp13 = tmp11 - tmp12
        tmp14 = tmp10 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/np/cnpitw4fc3ssbvinaev7p55xn6l36kf6wftyuwcma4u2bi7g27w3.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_161 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_161', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/g6/cg6vzzszn2soy7plc7pxf4lhtvz5acg3uv2ikxabsvfycjr7vuc5.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_162 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_162', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 112)
    x2 = xindex % 112
    y4 = yindex
    x5 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp9 = tl.load(in_ptr1 + (y0 + (96*x5) + (1204224*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0 + (96*x5) + (1204224*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = x3
    tmp1 = tl.full([1, 1], 113, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x2
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + (113*x3) + (12769*y4)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp10 = tmp8 * tmp9
    tmp13 = tmp11 - tmp12
    tmp15 = 9.964923469387754e-06
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(out_ptr0 + (x5 + (12544*y4)), tmp27, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ae/caehgcvdmgoyssgjzjutor4xhl4eg4dzxkso6b2qxgawx3nb5o5b.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_163 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_163', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/kh/ckhuux7wekfx3kkdg4yiks2inga7rij3m5wgkc3i2ekasahdzevg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_164 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_164', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/56/c56oy4ihqinkkj6t6cz4nrrbxnxt242ji2isgijqfu2q7k6pp6ye.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_165 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_165', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/vu/cvugmc2r2lvvnil57bko7rn4m7eoosbq3shsf6tm3vbsddjuheqr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_166 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_166', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ys/cysflovohfk23n4zn7rf6iiuvyxjypycz3rxgiouoospbnbtobsz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_167 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_167', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/l2/cl27wxudaw5o5pfamk64yfbds46o6r4phi3a47lsjzcbjt45shcy.py
# Source Nodes: [x_10], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_10 => mul_15, sigmoid_1
triton_red_fused_mul_silu_sum_168 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_168', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 98
    x1 = (xindex // 98) % 32
    x2 = (xindex // 3136)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (32*r3) + (4096*x0) + (401408*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/73/c73yhecjgsp46i5xcf7uidgdbptsdhctthpuqs5rankftar7r36g.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_10], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___0_____0___se_gate => sigmoid_3
# x_10 => mul_15, sigmoid_1
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_169 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_169', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tmp6 * tmp8
    tmp10 = tmp4 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fu/cfuus3bltx57el7mvl2y5i2qcsfiz5nn2t6yrh5xnh2atm5b6ejl.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_170 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_170', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/x3/cx3m7xlpfguugq7e3zgkwtys7ud7yvcgtk6cuqsttmwieegvw6gb.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_171 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_171', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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


# kernel path: /tmp/torchinductor_youkaichao/u3/cu3st3774yxhjgblvw7hc6vtrfa7b5t6r4rvf6gtr53pjhirzjk3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_172 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_172', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/3n/c3nahsy3uhslwe3i36sjwt7if5c2jju7u6yvsbv2uwf4l6cmuuzt.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___0_____0___se_gate => sigmoid_3
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_173 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_173', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((12544*x1) + (401408*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (32*((r2 + (128*x0)) // 12544))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (32*((r2 + (128*x0)) // 12544))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (32*r2) + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 12544.0
        tmp6 = tmp4 / tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12 + tmp10
        tmp14 = tmp9 * tmp13
        tmp15 = tmp7 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/23/c234p6wduxxjwc3orxr7wphff62pqojijm5pirn36kxmf6t2xhnk.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___0_____0___se_gate => sigmoid_3
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_174 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_174', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/2z/c2zpdpj3j6bgm6esl63z5dp33kem6forqx3zgw33p4dw23xrdl25.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___0_____0___se_gate => sigmoid_3
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_175 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_175', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((12544*x0) + (401408*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (32*((r2 + (128*x1)) // 12544))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (32*((r2 + (128*x1)) // 12544))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 12544.0
        tmp6 = tmp4 / tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12 + tmp10
        tmp14 = tmp9 * tmp13
        tmp15 = tmp7 * tmp14
        tmp18 = tmp16 - tmp17
        tmp19 = tmp15 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qb/cqbthilgepl2ms7isngcmlb6yohbrs5bfuvidm5mjxuow6txkswv.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___0_____0___se_gate => sigmoid_3
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_176 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_176', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/kk/ckkfbr6glwb26sdba2zh7f67banqbqtwrejeluhcyjtkoacexhqx.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___0_____0___se_gate => sigmoid_3
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_177 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_177', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100352
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 12544
    y1 = (yindex // 12544)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (12544*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (32*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (32*y1)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (32*y3)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (32*y3)), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 12544.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp7 * tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 9.964923469387754e-06
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (32*y3)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/56/c56qoxvw2zel6y77ul34kfvw7ulisbj5n4g6aws42mek35cbi6e6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_178 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_178', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (32*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (12544*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yv/cyvlmmbe3rzrsaggb3gox7o2pzw6bitygbcay3bw7zggrcnaopv6.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_179 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_179', 'mutated_arg_names': []}
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
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((12544*x1) + (401408*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (32*r2) + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7c/c7c4j34n3avfqosebiy4ilvlurh76dhoih7s6jaarqcnik6krq2j.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_180 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_180', 'mutated_arg_names': []}
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
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((12544*x0) + (401408*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/da/cdaxt562ksuidrwig35gmy7ksreafvat5e5rokxxc25aj5om5qi4.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_181 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_181', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 12544
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (32*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (32*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (12544*y3)), tmp19, xmask & ymask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_6, primals_8, primals_10, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_105, primals_107, primals_109, primals_110, primals_111, primals_113, primals_115, primals_116, primals_117, primals_118, primals_120, primals_122, primals_123, primals_124, primals_126, primals_128, primals_129, primals_130, primals_131, primals_133, primals_135, primals_136, primals_137, primals_139, primals_141, primals_142, primals_143, primals_144, primals_146, primals_148, primals_149, primals_150, primals_151, primals_153, primals_155, primals_156, primals_157, primals_158, primals_160, primals_162, primals_163, primals_164, primals_165, primals_167, primals_169, primals_170, primals_171, primals_172, primals_174, primals_176, primals_177, primals_178, primals_180, primals_182, primals_183, primals_184, primals_185, primals_187, primals_189, primals_190, primals_191, primals_192, primals_194, primals_196, primals_197, primals_198, primals_199, primals_201, primals_203, primals_204, primals_205, primals_206, primals_208, primals_210, primals_211, constant_pad_nd, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, add_9, mean, convolution_2, mul_16, convolution_3, mul_17, convolution_4, squeeze_7, add_14, convolution_5, squeeze_10, constant_pad_nd_1, convolution_6, squeeze_13, add_24, mean_1, convolution_7, mul_41, convolution_8, mul_42, convolution_9, squeeze_16, add_29, convolution_10, squeeze_19, mul_57, convolution_11, squeeze_22, add_39, mean_2, convolution_12, mul_66, convolution_13, mul_67, convolution_14, squeeze_25, add_45, convolution_15, squeeze_28, constant_pad_nd_2, convolution_16, squeeze_31, add_55, mean_3, convolution_17, mul_91, convolution_18, mul_92, convolution_19, squeeze_34, add_60, convolution_20, squeeze_37, mul_107, convolution_21, squeeze_40, add_70, mean_4, convolution_22, mul_116, convolution_23, mul_117, convolution_24, squeeze_43, add_76, convolution_25, squeeze_46, constant_pad_nd_3, convolution_26, squeeze_49, add_86, mean_5, convolution_27, mul_141, convolution_28, mul_142, convolution_29, squeeze_52, add_91, convolution_30, squeeze_55, mul_157, convolution_31, squeeze_58, add_101, mean_6, convolution_32, mul_166, convolution_33, mul_167, convolution_34, squeeze_61, add_107, convolution_35, squeeze_64, mul_182, convolution_36, squeeze_67, add_117, mean_7, convolution_37, mul_191, convolution_38, mul_192, convolution_39, squeeze_70, add_123, convolution_40, squeeze_73, mul_207, convolution_41, squeeze_76, add_133, mean_8, convolution_42, mul_216, convolution_43, mul_217, convolution_44, squeeze_79, add_138, convolution_45, squeeze_82, mul_232, convolution_46, squeeze_85, add_148, mean_9, convolution_47, mul_241, convolution_48, mul_242, convolution_49, squeeze_88, add_154, convolution_50, squeeze_91, mul_257, convolution_51, squeeze_94, add_164, mean_10, convolution_52, mul_266, convolution_53, mul_267, convolution_54, squeeze_97, add_170, convolution_55, squeeze_100, constant_pad_nd_4, convolution_56, squeeze_103, add_180, mean_11, convolution_57, mul_291, convolution_58, mul_292, convolution_59, squeeze_106, add_185, convolution_60, squeeze_109, mul_307, convolution_61, squeeze_112, add_195, mean_12, convolution_62, mul_316, convolution_63, mul_317, convolution_64, squeeze_115, add_201, convolution_65, squeeze_118, mul_332, convolution_66, squeeze_121, add_211, mean_13, convolution_67, mul_341, convolution_68, mul_342, convolution_69, squeeze_124, add_217, convolution_70, squeeze_127, mul_357, convolution_71, squeeze_130, add_227, mean_14, convolution_72, mul_366, convolution_73, mul_367, convolution_74, squeeze_133, add_233, convolution_75, squeeze_136, mul_382, convolution_76, squeeze_139, add_243, mean_15, convolution_77, mul_391, convolution_78, mul_392, convolution_79, squeeze_142, add_248, convolution_80, squeeze_145, view, permute_1, mul_409, unsqueeze_198, unsqueeze_210, unsqueeze_222, mul_449, unsqueeze_234, unsqueeze_246, unsqueeze_258, mul_489, unsqueeze_270, unsqueeze_282, unsqueeze_294, mul_529, unsqueeze_306, unsqueeze_318, unsqueeze_330, mul_569, unsqueeze_342, unsqueeze_354, unsqueeze_366, mul_609, unsqueeze_378, unsqueeze_390, unsqueeze_402, mul_649, unsqueeze_414, unsqueeze_426, unsqueeze_438, mul_689, unsqueeze_450, unsqueeze_462, unsqueeze_474, mul_729, unsqueeze_486, unsqueeze_498, unsqueeze_510, mul_769, unsqueeze_522, unsqueeze_534, unsqueeze_546, mul_809, unsqueeze_558, unsqueeze_570, unsqueeze_582, mul_849, unsqueeze_594, unsqueeze_606, unsqueeze_618, mul_889, unsqueeze_630, unsqueeze_642, unsqueeze_654, mul_929, unsqueeze_666, unsqueeze_678, unsqueeze_690, mul_969, unsqueeze_702, unsqueeze_714, unsqueeze_726, mul_1009, unsqueeze_738, unsqueeze_750, unsqueeze_762, mul_1049, unsqueeze_774, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_8, (96, ), (1, ))
    assert_size_stride(primals_10, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_11, (96, ), (1, ))
    assert_size_stride(primals_13, (24, ), (1, ))
    assert_size_stride(primals_15, (144, ), (1, ))
    assert_size_stride(primals_17, (144, ), (1, ))
    assert_size_stride(primals_19, (24, ), (1, ))
    assert_size_stride(primals_21, (144, ), (1, ))
    assert_size_stride(primals_23, (144, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_24, (144, ), (1, ))
    assert_size_stride(primals_26, (40, ), (1, ))
    assert_size_stride(primals_28, (240, ), (1, ))
    assert_size_stride(primals_30, (240, ), (1, ))
    assert_size_stride(primals_32, (40, ), (1, ))
    assert_size_stride(primals_34, (240, ), (1, ))
    assert_size_stride(primals_36, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_37, (240, ), (1, ))
    assert_size_stride(primals_39, (80, ), (1, ))
    assert_size_stride(primals_41, (480, ), (1, ))
    assert_size_stride(primals_43, (480, ), (1, ))
    assert_size_stride(primals_45, (80, ), (1, ))
    assert_size_stride(primals_47, (480, ), (1, ))
    assert_size_stride(primals_49, (480, ), (1, ))
    assert_size_stride(primals_51, (80, ), (1, ))
    assert_size_stride(primals_53, (480, ), (1, ))
    assert_size_stride(primals_55, (480, ), (1, ))
    assert_size_stride(primals_57, (112, ), (1, ))
    assert_size_stride(primals_59, (672, ), (1, ))
    assert_size_stride(primals_61, (672, ), (1, ))
    assert_size_stride(primals_63, (112, ), (1, ))
    assert_size_stride(primals_65, (672, ), (1, ))
    assert_size_stride(primals_67, (672, ), (1, ))
    assert_size_stride(primals_69, (112, ), (1, ))
    assert_size_stride(primals_71, (672, ), (1, ))
    assert_size_stride(primals_73, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_74, (672, ), (1, ))
    assert_size_stride(primals_76, (192, ), (1, ))
    assert_size_stride(primals_78, (1152, ), (1, ))
    assert_size_stride(primals_80, (1152, ), (1, ))
    assert_size_stride(primals_82, (192, ), (1, ))
    assert_size_stride(primals_84, (1152, ), (1, ))
    assert_size_stride(primals_86, (1152, ), (1, ))
    assert_size_stride(primals_88, (192, ), (1, ))
    assert_size_stride(primals_90, (1152, ), (1, ))
    assert_size_stride(primals_92, (1152, ), (1, ))
    assert_size_stride(primals_94, (192, ), (1, ))
    assert_size_stride(primals_96, (1152, ), (1, ))
    assert_size_stride(primals_98, (1152, ), (1, ))
    assert_size_stride(primals_100, (320, ), (1, ))
    assert_size_stride(primals_102, (1280, ), (1, ))
    assert_size_stride(primals_104, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_105, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_107, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_109, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_110, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_111, (4, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_113, (96, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_115, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_116, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_117, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_118, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_120, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_122, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_123, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_124, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_126, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_128, (40, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_129, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_130, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_131, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_133, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(primals_135, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_136, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_137, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_139, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(primals_141, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_142, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_143, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_144, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_146, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_148, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_149, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_150, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_151, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_153, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_155, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_156, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_157, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_158, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_160, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_162, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_163, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_164, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_165, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_167, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_169, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_170, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_171, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_172, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_174, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_176, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_177, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_178, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_180, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_182, (192, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_183, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_184, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_185, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_187, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_189, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_190, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_191, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_192, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_194, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_196, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_197, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_198, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_199, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_201, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_203, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_204, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_205, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_206, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_208, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_210, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_211, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(constant_pad_nd, (8, 3, 225, 225), (151875, 1, 675, 3))
    assert_size_stride(convolution, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(mul_7, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_1, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(add_9, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(mean, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(convolution_2, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(mul_16, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(convolution_3, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(mul_17, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_4, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_7, (16, ), (1, ))
    assert_size_stride(add_14, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_5, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(squeeze_10, (96, ), (1, ))
    assert_size_stride(constant_pad_nd_1, (8, 96, 113, 113), (1225824, 1, 10848, 96))
    assert_size_stride(convolution_6, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(squeeze_13, (96, ), (1, ))
    assert_size_stride(add_24, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(mean_1, (8, 96, 1, 1), (96, 1, 96, 96))
    assert_size_stride(convolution_7, (8, 4, 1, 1), (4, 1, 4, 4))
    assert_size_stride(mul_41, (8, 4, 1, 1), (4, 1, 4, 4))
    assert_size_stride(convolution_8, (8, 96, 1, 1), (96, 1, 96, 96))
    assert_size_stride(mul_42, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(convolution_9, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_16, (24, ), (1, ))
    assert_size_stride(add_29, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_10, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(squeeze_19, (144, ), (1, ))
    assert_size_stride(mul_57, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(convolution_11, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(squeeze_22, (144, ), (1, ))
    assert_size_stride(add_39, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(mean_2, (8, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(convolution_12, (8, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(mul_66, (8, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(convolution_13, (8, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(mul_67, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(convolution_14, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_25, (24, ), (1, ))
    assert_size_stride(add_45, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_15, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(squeeze_28, (144, ), (1, ))
    assert_size_stride(constant_pad_nd_2, (8, 144, 59, 59), (501264, 1, 8496, 144))
    assert_size_stride(convolution_16, (8, 144, 28, 28), (112896, 1, 4032, 144))
    assert_size_stride(squeeze_31, (144, ), (1, ))
    assert_size_stride(add_55, (8, 144, 28, 28), (112896, 1, 4032, 144))
    assert_size_stride(mean_3, (8, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(convolution_17, (8, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(mul_91, (8, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(convolution_18, (8, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(mul_92, (8, 144, 28, 28), (112896, 1, 4032, 144))
    assert_size_stride(convolution_19, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_34, (40, ), (1, ))
    assert_size_stride(add_60, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_20, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(squeeze_37, (240, ), (1, ))
    assert_size_stride(mul_107, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(convolution_21, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(squeeze_40, (240, ), (1, ))
    assert_size_stride(add_70, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(mean_4, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(convolution_22, (8, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(mul_116, (8, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(convolution_23, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(mul_117, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(convolution_24, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_43, (40, ), (1, ))
    assert_size_stride(add_76, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_25, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(squeeze_46, (240, ), (1, ))
    assert_size_stride(constant_pad_nd_3, (8, 240, 29, 29), (201840, 1, 6960, 240))
    assert_size_stride(convolution_26, (8, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(squeeze_49, (240, ), (1, ))
    assert_size_stride(add_86, (8, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(mean_5, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(convolution_27, (8, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(mul_141, (8, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(convolution_28, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(mul_142, (8, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(convolution_29, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(squeeze_52, (80, ), (1, ))
    assert_size_stride(add_91, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_30, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_55, (480, ), (1, ))
    assert_size_stride(mul_157, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_31, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_58, (480, ), (1, ))
    assert_size_stride(add_101, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_6, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_32, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(mul_166, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(convolution_33, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_167, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_34, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(squeeze_61, (80, ), (1, ))
    assert_size_stride(add_107, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_35, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_64, (480, ), (1, ))
    assert_size_stride(mul_182, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_36, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_67, (480, ), (1, ))
    assert_size_stride(add_117, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_7, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_37, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(mul_191, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(convolution_38, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_192, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_39, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(squeeze_70, (80, ), (1, ))
    assert_size_stride(add_123, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_40, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_73, (480, ), (1, ))
    assert_size_stride(mul_207, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_41, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_76, (480, ), (1, ))
    assert_size_stride(add_133, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_8, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_42, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(mul_216, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(convolution_43, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_217, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_44, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(squeeze_79, (112, ), (1, ))
    assert_size_stride(add_138, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(convolution_45, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(squeeze_82, (672, ), (1, ))
    assert_size_stride(mul_232, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_46, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(squeeze_85, (672, ), (1, ))
    assert_size_stride(add_148, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(mean_9, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(convolution_47, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_241, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_48, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_242, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_49, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(squeeze_88, (112, ), (1, ))
    assert_size_stride(add_154, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(convolution_50, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(squeeze_91, (672, ), (1, ))
    assert_size_stride(mul_257, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_51, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(squeeze_94, (672, ), (1, ))
    assert_size_stride(add_164, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(mean_10, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(convolution_52, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_266, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_53, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_267, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_54, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(squeeze_97, (112, ), (1, ))
    assert_size_stride(add_170, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(convolution_55, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(squeeze_100, (672, ), (1, ))
    assert_size_stride(constant_pad_nd_4, (8, 672, 17, 17), (194208, 1, 11424, 672))
    assert_size_stride(convolution_56, (8, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(squeeze_103, (672, ), (1, ))
    assert_size_stride(add_180, (8, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(mean_11, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(convolution_57, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_291, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_58, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_292, (8, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(convolution_59, (8, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(squeeze_106, (192, ), (1, ))
    assert_size_stride(add_185, (8, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_60, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(squeeze_109, (1152, ), (1, ))
    assert_size_stride(mul_307, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_61, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(squeeze_112, (1152, ), (1, ))
    assert_size_stride(add_195, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mean_12, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_62, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_316, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_63, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_317, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_64, (8, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(squeeze_115, (192, ), (1, ))
    assert_size_stride(add_201, (8, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_65, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(squeeze_118, (1152, ), (1, ))
    assert_size_stride(mul_332, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_66, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(squeeze_121, (1152, ), (1, ))
    assert_size_stride(add_211, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mean_13, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_67, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_341, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_68, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_342, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_69, (8, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(squeeze_124, (192, ), (1, ))
    assert_size_stride(add_217, (8, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_70, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(squeeze_127, (1152, ), (1, ))
    assert_size_stride(mul_357, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_71, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(squeeze_130, (1152, ), (1, ))
    assert_size_stride(add_227, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mean_14, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_72, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_366, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_73, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_367, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_74, (8, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(squeeze_133, (192, ), (1, ))
    assert_size_stride(add_233, (8, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_75, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(squeeze_136, (1152, ), (1, ))
    assert_size_stride(mul_382, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_76, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(squeeze_139, (1152, ), (1, ))
    assert_size_stride(add_243, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mean_15, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_77, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_391, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_78, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_392, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_79, (8, 320, 7, 7), (15680, 1, 2240, 320))
    assert_size_stride(squeeze_142, (320, ), (1, ))
    assert_size_stride(add_248, (8, 320, 7, 7), (15680, 1, 2240, 320))
    assert_size_stride(convolution_80, (8, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(squeeze_145, (1280, ), (1, ))
    assert_size_stride(view, (8, 1280), (1280, 1))
    assert_size_stride(permute_1, (1000, 1280), (1280, 1))
    assert_size_stride(mul_409, (8, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(unsqueeze_198, (1, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(unsqueeze_210, (1, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(unsqueeze_222, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(mul_449, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(unsqueeze_234, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_246, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(mul_489, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(unsqueeze_270, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_282, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_294, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(mul_529, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(unsqueeze_306, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_318, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_330, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(mul_569, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(unsqueeze_342, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_354, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_366, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(mul_609, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(unsqueeze_378, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_390, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(mul_649, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(unsqueeze_414, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_426, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(unsqueeze_438, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(mul_689, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(unsqueeze_450, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_462, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_729, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(unsqueeze_486, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_498, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_510, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_769, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(unsqueeze_522, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_534, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_546, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_809, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(unsqueeze_558, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_570, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_582, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(mul_849, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(unsqueeze_594, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(unsqueeze_606, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(unsqueeze_618, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(mul_889, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(unsqueeze_630, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(unsqueeze_642, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(unsqueeze_654, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(mul_929, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(unsqueeze_666, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(unsqueeze_678, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_690, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(mul_969, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(unsqueeze_702, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(unsqueeze_714, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_726, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(mul_1009, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(unsqueeze_738, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_750, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_762, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(mul_1049, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(unsqueeze_774, (1, 32, 1, 1), (32, 1, 1, 1))
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
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_div_mul_native_batch_norm_backward_1.run(buf0, mul_409, convolution_80, unsqueeze_198, buf3, buf5, 5120, 98, grid=grid(5120), stream=stream0)
        buf4 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_mul_native_batch_norm_backward_2.run(buf3, buf4, 1280, 4, grid=grid(1280), stream=stream0)
        del buf3
        buf6 = empty((1280, ), device='cuda', dtype=torch.float32)
        buf7 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_mul_native_batch_norm_backward_3.run(buf5, squeeze_145, buf6, buf7, 1280, 4, grid=grid(1280), stream=stream0)
        del buf5
        buf8 = empty((8, 1280, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_div_mul_native_batch_norm_backward_4.run(buf0, mul_409, convolution_80, unsqueeze_198, buf6, squeeze_145, buf4, primals_102, buf8, 392, 1280, grid=grid(392, 1280), stream=stream0)
        del buf0
        del convolution_80
        del mul_409
        del primals_102
        del squeeze_145
        del unsqueeze_198
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
        buf9 = aten.convolution_backward(buf8, add_248, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_248
        del buf8
        del primals_211
        buf10 = buf9[0]
        buf11 = buf9[1]
        del buf9
        buf12 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_5.run(buf10, buf12, 320, 392, grid=grid(320), stream=stream0)
        buf13 = reinterpret_tensor(buf6, (320, 4), (1, 320), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_6.run(buf10, convolution_79, unsqueeze_210, buf13, 1280, 98, grid=grid(1280), stream=stream0)
        buf14 = empty((320, ), device='cuda', dtype=torch.float32)
        buf15 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_7.run(buf13, squeeze_142, buf14, buf15, 320, 4, grid=grid(320), stream=stream0)
        del buf13
        buf16 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_8.run(buf16, convolution_79, unsqueeze_210, buf14, squeeze_142, buf12, primals_100, 2560, 49, grid=grid(2560, 49), stream=stream0)
        del buf14
        del convolution_79
        del primals_100
        del squeeze_142
        del unsqueeze_210
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf17 = aten.convolution_backward(buf16, mul_392, primals_210, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_392
        del primals_210
        buf18 = buf17[0]
        buf19 = buf17[1]
        del buf17
        buf20 = empty_strided((8, 1152, 1, 1), (1152, 1, 9216, 9216), device='cuda', dtype=torch.float32)
        buf21 = reinterpret_tensor(buf20, (8, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf20  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate, x_267], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_9.run(buf21, buf18, add_243, convolution_78, 9216, 49, grid=grid(9216), stream=stream0)
        buf22 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf21, buf22, 1152, 8, grid=grid(1152), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf23 = aten.convolution_backward(buf21, mul_391, primals_208, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf21
        del mul_391
        del primals_208
        buf24 = buf23[0]
        buf25 = buf23[1]
        del buf23
        buf26 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_11.run(buf26, convolution_77, 384, grid=grid(384), stream=stream0)
        del convolution_77
        buf27 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf26, buf27, 48, 8, grid=grid(48), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf28 = aten.convolution_backward(buf26, mean_15, primals_206, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf26
        del mean_15
        del primals_206
        buf29 = buf28[0]
        buf30 = buf28[1]
        del buf28
        buf31 = empty_strided((1152, 4), (1, 1152), device='cuda', dtype=torch.float32)
        buf33 = empty_strided((1152, 4), (1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_13.run(buf18, convolution_78, buf29, add_243, convolution_76, unsqueeze_222, buf31, buf33, 4608, 98, grid=grid(4608), stream=stream0)
        buf32 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14.run(buf31, buf32, 1152, 4, grid=grid(1152), stream=stream0)
        buf34 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf36 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15.run(buf33, squeeze_139, buf34, buf36, 1152, 4, grid=grid(1152), stream=stream0)
        buf35 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_16.run(buf18, convolution_78, buf29, add_243, convolution_76, unsqueeze_222, buf34, squeeze_139, buf32, buf35, 392, 1152, grid=grid(392, 1152), stream=stream0)
        del add_243
        del convolution_76
        del convolution_78
        del unsqueeze_222
        buf37 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_17.run(buf35, squeeze_139, primals_98, buf37, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del buf35
        del primals_98
        del squeeze_139
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf38 = aten.convolution_backward(buf37, mul_382, primals_205, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1152, [True, True, False])
        del buf37
        del mul_382
        del primals_205
        buf39 = buf38[0]
        buf40 = buf38[1]
        del buf38
        buf41 = buf33; del buf33  # reuse
        buf43 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_18.run(buf39, mul_449, convolution_75, unsqueeze_234, buf41, buf43, 4608, 98, grid=grid(4608), stream=stream0)
        buf42 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14.run(buf41, buf42, 1152, 4, grid=grid(1152), stream=stream0)
        buf44 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf45 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15.run(buf43, squeeze_136, buf44, buf45, 1152, 4, grid=grid(1152), stream=stream0)
        buf46 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_19.run(buf46, mul_449, convolution_75, unsqueeze_234, buf44, squeeze_136, buf42, primals_96, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del convolution_75
        del mul_449
        del primals_96
        del squeeze_136
        del unsqueeze_234
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf47 = aten.convolution_backward(buf46, add_233, primals_204, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_233
        del primals_204
        buf48 = buf47[0]
        buf49 = buf47[1]
        del buf47
        buf50 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_20.run(buf48, buf50, 192, 392, grid=grid(192), stream=stream0)
        buf51 = empty_strided((192, 4), (1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_21.run(buf48, convolution_74, unsqueeze_246, buf51, 768, 98, grid=grid(768), stream=stream0)
        buf52 = empty((192, ), device='cuda', dtype=torch.float32)
        buf53 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_22.run(buf51, squeeze_133, buf52, buf53, 192, 4, grid=grid(192), stream=stream0)
        buf54 = empty((8, 192, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_23.run(buf48, convolution_74, unsqueeze_246, buf52, squeeze_133, buf50, primals_94, buf54, 1536, 49, grid=grid(1536, 49), stream=stream0)
        del convolution_74
        del primals_94
        del squeeze_133
        del unsqueeze_246
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf55 = aten.convolution_backward(buf54, mul_367, primals_203, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_367
        del primals_203
        buf56 = buf55[0]
        buf57 = buf55[1]
        del buf55
        buf58 = reinterpret_tensor(buf29, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf29  # reuse
        buf59 = reinterpret_tensor(buf58, (8, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf58  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_250], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_9.run(buf59, buf56, add_227, convolution_73, 9216, 49, grid=grid(9216), stream=stream0)
        buf60 = buf44; del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf59, buf60, 1152, 8, grid=grid(1152), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf61 = aten.convolution_backward(buf59, mul_366, primals_201, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf59
        del mul_366
        del primals_201
        buf62 = buf61[0]
        buf63 = buf61[1]
        del buf61
        buf64 = buf62; del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_11.run(buf64, convolution_72, 384, grid=grid(384), stream=stream0)
        del convolution_72
        buf65 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf64, buf65, 48, 8, grid=grid(48), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf66 = aten.convolution_backward(buf64, mean_14, primals_199, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf64
        del mean_14
        del primals_199
        buf67 = buf66[0]
        buf68 = buf66[1]
        del buf66
        buf69 = buf43; del buf43  # reuse
        buf71 = buf41; del buf41  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_13.run(buf56, convolution_73, buf67, add_227, convolution_71, unsqueeze_258, buf69, buf71, 4608, 98, grid=grid(4608), stream=stream0)
        buf70 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14.run(buf69, buf70, 1152, 4, grid=grid(1152), stream=stream0)
        buf72 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf74 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15.run(buf71, squeeze_130, buf72, buf74, 1152, 4, grid=grid(1152), stream=stream0)
        buf73 = reinterpret_tensor(buf46, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf46  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_16.run(buf56, convolution_73, buf67, add_227, convolution_71, unsqueeze_258, buf72, squeeze_130, buf70, buf73, 392, 1152, grid=grid(392, 1152), stream=stream0)
        del add_227
        del convolution_71
        del convolution_73
        del unsqueeze_258
        buf75 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_17.run(buf73, squeeze_130, primals_92, buf75, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del buf73
        del primals_92
        del squeeze_130
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf76 = aten.convolution_backward(buf75, mul_357, primals_198, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
        del buf75
        del mul_357
        del primals_198
        buf77 = buf76[0]
        buf78 = buf76[1]
        del buf76
        buf79 = buf71; del buf71  # reuse
        buf81 = buf69; del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_18.run(buf77, mul_489, convolution_70, unsqueeze_270, buf79, buf81, 4608, 98, grid=grid(4608), stream=stream0)
        buf80 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14.run(buf79, buf80, 1152, 4, grid=grid(1152), stream=stream0)
        buf82 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf83 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15.run(buf81, squeeze_127, buf82, buf83, 1152, 4, grid=grid(1152), stream=stream0)
        buf84 = buf77; del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_19.run(buf84, mul_489, convolution_70, unsqueeze_270, buf82, squeeze_127, buf80, primals_90, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del convolution_70
        del mul_489
        del primals_90
        del squeeze_127
        del unsqueeze_270
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf85 = aten.convolution_backward(buf84, add_217, primals_197, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_217
        del primals_197
        buf86 = buf85[0]
        buf87 = buf85[1]
        del buf85
        buf88 = buf52; del buf52  # reuse
        buf89 = empty((192, ), device='cuda', dtype=torch.float32)
        buf90 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_24.run(buf48, buf86, convolution_69, unsqueeze_282, squeeze_124, buf88, buf89, buf90, 192, 392, grid=grid(192), stream=stream0)
        buf91 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_25.run(buf48, buf86, convolution_69, unsqueeze_282, buf89, squeeze_124, buf88, primals_88, buf91, 1536, 49, grid=grid(1536, 49), stream=stream0)
        del convolution_69
        del primals_88
        del squeeze_124
        del unsqueeze_282
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf92 = aten.convolution_backward(buf91, mul_342, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_342
        del primals_196
        buf93 = buf92[0]
        buf94 = buf92[1]
        del buf92
        buf95 = reinterpret_tensor(buf67, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf67  # reuse
        buf96 = reinterpret_tensor(buf95, (8, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf95  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_233], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_9.run(buf96, buf93, add_211, convolution_68, 9216, 49, grid=grid(9216), stream=stream0)
        buf97 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf96, buf97, 1152, 8, grid=grid(1152), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf98 = aten.convolution_backward(buf96, mul_341, primals_194, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf96
        del mul_341
        del primals_194
        buf99 = buf98[0]
        buf100 = buf98[1]
        del buf98
        buf101 = buf99; del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_11.run(buf101, convolution_67, 384, grid=grid(384), stream=stream0)
        del convolution_67
        buf102 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf101, buf102, 48, 8, grid=grid(48), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf103 = aten.convolution_backward(buf101, mean_13, primals_192, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf101
        del mean_13
        del primals_192
        buf104 = buf103[0]
        buf105 = buf103[1]
        del buf103
        buf106 = buf81; del buf81  # reuse
        buf108 = buf79; del buf79  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_13.run(buf93, convolution_68, buf104, add_211, convolution_66, unsqueeze_294, buf106, buf108, 4608, 98, grid=grid(4608), stream=stream0)
        buf107 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14.run(buf106, buf107, 1152, 4, grid=grid(1152), stream=stream0)
        buf109 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf111 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15.run(buf108, squeeze_121, buf109, buf111, 1152, 4, grid=grid(1152), stream=stream0)
        buf110 = reinterpret_tensor(buf84, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf84  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_16.run(buf93, convolution_68, buf104, add_211, convolution_66, unsqueeze_294, buf109, squeeze_121, buf107, buf110, 392, 1152, grid=grid(392, 1152), stream=stream0)
        del add_211
        del convolution_66
        del convolution_68
        del unsqueeze_294
        buf112 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_17.run(buf110, squeeze_121, primals_86, buf112, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del buf110
        del primals_86
        del squeeze_121
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf113 = aten.convolution_backward(buf112, mul_332, primals_191, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
        del buf112
        del mul_332
        del primals_191
        buf114 = buf113[0]
        buf115 = buf113[1]
        del buf113
        buf116 = buf108; del buf108  # reuse
        buf118 = buf106; del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_18.run(buf114, mul_529, convolution_65, unsqueeze_306, buf116, buf118, 4608, 98, grid=grid(4608), stream=stream0)
        buf117 = buf109; del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14.run(buf116, buf117, 1152, 4, grid=grid(1152), stream=stream0)
        buf119 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf120 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15.run(buf118, squeeze_118, buf119, buf120, 1152, 4, grid=grid(1152), stream=stream0)
        buf121 = buf114; del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_19.run(buf121, mul_529, convolution_65, unsqueeze_306, buf119, squeeze_118, buf117, primals_84, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del convolution_65
        del mul_529
        del primals_84
        del squeeze_118
        del unsqueeze_306
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf122 = aten.convolution_backward(buf121, add_201, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_201
        del primals_190
        buf123 = buf122[0]
        buf124 = buf122[1]
        del buf122
        buf125 = buf89; del buf89  # reuse
        buf126 = empty((192, ), device='cuda', dtype=torch.float32)
        buf128 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_26.run(buf48, buf86, buf123, convolution_64, unsqueeze_318, squeeze_115, buf125, buf126, buf128, 192, 392, grid=grid(192), stream=stream0)
        buf127 = buf91; del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_27.run(buf48, buf86, buf123, convolution_64, unsqueeze_318, buf126, squeeze_115, buf125, primals_82, buf127, 1536, 49, grid=grid(1536, 49), stream=stream0)
        del convolution_64
        del primals_82
        del squeeze_115
        del unsqueeze_318
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf129 = aten.convolution_backward(buf127, mul_317, primals_189, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf127
        del mul_317
        del primals_189
        buf130 = buf129[0]
        buf131 = buf129[1]
        del buf129
        buf132 = reinterpret_tensor(buf104, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf104  # reuse
        buf133 = reinterpret_tensor(buf132, (8, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf132  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_216], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_9.run(buf133, buf130, add_195, convolution_63, 9216, 49, grid=grid(9216), stream=stream0)
        buf134 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf133, buf134, 1152, 8, grid=grid(1152), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf135 = aten.convolution_backward(buf133, mul_316, primals_187, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf133
        del mul_316
        del primals_187
        buf136 = buf135[0]
        buf137 = buf135[1]
        del buf135
        buf138 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_11.run(buf138, convolution_62, 384, grid=grid(384), stream=stream0)
        del convolution_62
        buf139 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf138, buf139, 48, 8, grid=grid(48), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf140 = aten.convolution_backward(buf138, mean_12, primals_185, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf138
        del mean_12
        del primals_185
        buf141 = buf140[0]
        buf142 = buf140[1]
        del buf140
        buf143 = buf118; del buf118  # reuse
        buf145 = buf116; del buf116  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_13.run(buf130, convolution_63, buf141, add_195, convolution_61, unsqueeze_330, buf143, buf145, 4608, 98, grid=grid(4608), stream=stream0)
        buf144 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14.run(buf143, buf144, 1152, 4, grid=grid(1152), stream=stream0)
        buf146 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf148 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15.run(buf145, squeeze_112, buf146, buf148, 1152, 4, grid=grid(1152), stream=stream0)
        buf147 = reinterpret_tensor(buf121, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf121  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_16.run(buf130, convolution_63, buf141, add_195, convolution_61, unsqueeze_330, buf146, squeeze_112, buf144, buf147, 392, 1152, grid=grid(392, 1152), stream=stream0)
        del add_195
        del buf141
        del convolution_61
        del convolution_63
        del unsqueeze_330
        buf149 = buf130; del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_17.run(buf147, squeeze_112, primals_80, buf149, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del buf147
        del primals_80
        del squeeze_112
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf150 = aten.convolution_backward(buf149, mul_307, primals_184, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
        del buf149
        del mul_307
        del primals_184
        buf151 = buf150[0]
        buf152 = buf150[1]
        del buf150
        buf153 = buf145; del buf145  # reuse
        buf155 = buf143; del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_18.run(buf151, mul_569, convolution_60, unsqueeze_342, buf153, buf155, 4608, 98, grid=grid(4608), stream=stream0)
        buf154 = buf146; del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14.run(buf153, buf154, 1152, 4, grid=grid(1152), stream=stream0)
        del buf153
        buf156 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf157 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15.run(buf155, squeeze_109, buf156, buf157, 1152, 4, grid=grid(1152), stream=stream0)
        del buf155
        buf158 = buf151; del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_19.run(buf158, mul_569, convolution_60, unsqueeze_342, buf156, squeeze_109, buf154, primals_78, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del convolution_60
        del mul_569
        del primals_78
        del squeeze_109
        del unsqueeze_342
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf159 = aten.convolution_backward(buf158, add_185, primals_183, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_185
        del buf158
        del primals_183
        buf160 = buf159[0]
        buf161 = buf159[1]
        del buf159
        buf162 = buf126; del buf126  # reuse
        buf163 = empty((192, ), device='cuda', dtype=torch.float32)
        buf165 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_28.run(buf48, buf86, buf123, buf160, convolution_59, unsqueeze_354, squeeze_106, buf162, buf163, buf165, 192, 392, grid=grid(192), stream=stream0)
        buf164 = buf123; del buf123  # reuse
        buf166 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_29.run(buf166, buf48, buf86, buf160, convolution_59, unsqueeze_354, buf163, squeeze_106, buf162, primals_76, 1536, 49, grid=grid(1536, 49), stream=stream0)
        del buf160
        del buf163
        del buf48
        del buf86
        del convolution_59
        del primals_76
        del squeeze_106
        del unsqueeze_354
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf167 = aten.convolution_backward(buf166, mul_292, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_292
        del primals_182
        buf168 = buf167[0]
        buf169 = buf167[1]
        del buf167
        buf170 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cuda', dtype=torch.float32)
        buf171 = reinterpret_tensor(buf170, (8, 672, 1, 1), (672, 1, 1, 1), 0); del buf170  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_200], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_30.run(buf171, buf168, add_180, convolution_58, 5376, 49, grid=grid(5376), stream=stream0)
        buf172 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_31.run(buf171, buf172, 672, 8, grid=grid(672), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf173 = aten.convolution_backward(buf171, mul_291, primals_180, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf171
        del mul_291
        del primals_180
        buf174 = buf173[0]
        buf175 = buf173[1]
        del buf173
        buf176 = buf174; del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_32.run(buf176, convolution_57, 224, grid=grid(224), stream=stream0)
        del convolution_57
        buf177 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_33.run(buf176, buf177, 28, 8, grid=grid(28), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf178 = aten.convolution_backward(buf176, mean_11, primals_178, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf176
        del mean_11
        del primals_178
        buf179 = buf178[0]
        buf180 = buf178[1]
        del buf178
        buf181 = empty_strided((672, 4), (1, 672), device='cuda', dtype=torch.float32)
        buf183 = empty_strided((672, 4), (1, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_34.run(buf168, convolution_58, buf179, add_180, convolution_56, unsqueeze_366, buf181, buf183, 2688, 98, grid=grid(2688), stream=stream0)
        buf182 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_35.run(buf181, buf182, 672, 4, grid=grid(672), stream=stream0)
        del buf181
        buf184 = empty((672, ), device='cuda', dtype=torch.float32)
        buf186 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_36.run(buf183, squeeze_103, buf184, buf186, 672, 4, grid=grid(672), stream=stream0)
        del buf183
        buf185 = empty_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_37.run(buf168, convolution_58, buf179, add_180, convolution_56, unsqueeze_366, buf184, squeeze_103, buf182, buf185, 392, 672, grid=grid(392, 672), stream=stream0)
        del add_180
        del convolution_56
        del convolution_58
        del unsqueeze_366
        buf187 = buf168; del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_38.run(buf185, squeeze_103, primals_74, buf187, 5376, 49, grid=grid(5376, 49), stream=stream0)
        del buf185
        del primals_74
        del squeeze_103
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf188 = aten.convolution_backward(buf187, constant_pad_nd_4, primals_73, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 672, [True, True, False])
        del buf187
        del constant_pad_nd_4
        del primals_73
        buf189 = buf188[0]
        buf190 = buf188[1]
        del buf188
        buf191 = empty((672, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_39.run(buf189, mul_609, buf191, 8736, 121, grid=grid(8736), stream=stream0)
        buf192 = buf184; del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_40.run(buf191, buf192, 672, 13, grid=grid(672), stream=stream0)
        buf193 = reinterpret_tensor(buf191, (672, 13), (1, 672), 0); del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_41.run(buf189, mul_609, convolution_55, unsqueeze_378, buf193, 8736, 121, grid=grid(8736), stream=stream0)
        buf194 = empty((672, ), device='cuda', dtype=torch.float32)
        buf195 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_42.run(buf193, squeeze_100, buf194, buf195, 672, 13, grid=grid(672), stream=stream0)
        buf196 = empty((8, 672, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_43.run(buf189, mul_609, convolution_55, unsqueeze_378, buf194, squeeze_100, buf192, primals_71, buf196, 5376, 196, grid=grid(5376, 196), stream=stream0)
        del buf189
        del convolution_55
        del mul_609
        del primals_71
        del squeeze_100
        del unsqueeze_378
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf197 = aten.convolution_backward(buf196, add_170, primals_177, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_170
        del primals_177
        buf198 = buf197[0]
        buf199 = buf197[1]
        del buf197
        buf200 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_44.run(buf198, buf200, 112, 1568, grid=grid(112), stream=stream0)
        buf201 = empty((112, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_45.run(buf198, convolution_54, unsqueeze_390, buf201, 1456, 121, grid=grid(1456), stream=stream0)
        buf202 = empty((112, ), device='cuda', dtype=torch.float32)
        buf203 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_46.run(buf201, squeeze_97, buf202, buf203, 112, 13, grid=grid(112), stream=stream0)
        del buf201
        buf204 = empty((8, 112, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_47.run(buf198, convolution_54, unsqueeze_390, buf202, squeeze_97, buf200, primals_69, buf204, 896, 196, grid=grid(896, 196), stream=stream0)
        del convolution_54
        del primals_69
        del squeeze_97
        del unsqueeze_390
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf205 = aten.convolution_backward(buf204, mul_267, primals_176, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_267
        del primals_176
        buf206 = buf205[0]
        buf207 = buf205[1]
        del buf205
        buf208 = empty_strided((8, 672, 1, 1, 2), (1344, 2, 10752, 10752, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_181], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_48.run(buf206, add_164, buf208, 10752, 98, grid=grid(10752), stream=stream0)
        buf209 = reinterpret_tensor(buf179, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf179  # reuse
        buf210 = reinterpret_tensor(buf209, (8, 672, 1, 1), (672, 1, 1, 1), 0); del buf209  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_181], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_49.run(buf210, buf208, convolution_53, 5376, 2, grid=grid(5376), stream=stream0)
        buf211 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_31.run(buf210, buf211, 672, 8, grid=grid(672), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf212 = aten.convolution_backward(buf210, mul_266, primals_174, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf210
        del mul_266
        del primals_174
        buf213 = buf212[0]
        buf214 = buf212[1]
        del buf212
        buf215 = buf213; del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_32.run(buf215, convolution_52, 224, grid=grid(224), stream=stream0)
        del convolution_52
        buf216 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_33.run(buf215, buf216, 28, 8, grid=grid(28), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf217 = aten.convolution_backward(buf215, mean_10, primals_172, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf215
        del mean_10
        del primals_172
        buf218 = buf217[0]
        buf219 = buf217[1]
        del buf217
        buf220 = reinterpret_tensor(buf193, (672, 13), (13, 1), 0); del buf193  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_50.run(buf206, convolution_53, buf218, add_164, buf220, 8736, 121, grid=grid(8736), stream=stream0)
        buf221 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_40.run(buf220, buf221, 672, 13, grid=grid(672), stream=stream0)
        buf222 = reinterpret_tensor(buf220, (672, 13), (1, 672), 0); del buf220  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_51.run(buf206, convolution_53, buf218, add_164, convolution_51, unsqueeze_402, buf222, 8736, 121, grid=grid(8736), stream=stream0)
        buf223 = empty((672, ), device='cuda', dtype=torch.float32)
        buf225 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_42.run(buf222, squeeze_94, buf223, buf225, 672, 13, grid=grid(672), stream=stream0)
        buf224 = reinterpret_tensor(buf196, (8, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf196  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_52.run(buf206, convolution_53, buf218, add_164, convolution_51, unsqueeze_402, buf223, squeeze_94, buf221, buf224, 1568, 672, grid=grid(1568, 672), stream=stream0)
        del add_164
        del convolution_51
        del convolution_53
        del unsqueeze_402
        buf226 = buf206; del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_53.run(buf224, squeeze_94, primals_67, buf226, 5376, 196, grid=grid(5376, 196), stream=stream0)
        del buf224
        del primals_67
        del squeeze_94
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf227 = aten.convolution_backward(buf226, mul_257, primals_171, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
        del buf226
        del mul_257
        del primals_171
        buf228 = buf227[0]
        buf229 = buf227[1]
        del buf227
        buf230 = reinterpret_tensor(buf222, (672, 13), (13, 1), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_54.run(buf228, mul_649, buf230, 8736, 121, grid=grid(8736), stream=stream0)
        buf231 = buf223; del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_40.run(buf230, buf231, 672, 13, grid=grid(672), stream=stream0)
        buf232 = reinterpret_tensor(buf230, (672, 13), (1, 672), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_55.run(buf228, mul_649, convolution_50, unsqueeze_414, buf232, 8736, 121, grid=grid(8736), stream=stream0)
        buf233 = empty((672, ), device='cuda', dtype=torch.float32)
        buf234 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_42.run(buf232, squeeze_91, buf233, buf234, 672, 13, grid=grid(672), stream=stream0)
        buf235 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_56.run(buf235, mul_649, convolution_50, unsqueeze_414, buf233, squeeze_91, buf231, primals_65, 5376, 196, grid=grid(5376, 196), stream=stream0)
        del convolution_50
        del mul_649
        del primals_65
        del squeeze_91
        del unsqueeze_414
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf236 = aten.convolution_backward(buf235, add_154, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_154
        del primals_170
        buf237 = buf236[0]
        buf238 = buf236[1]
        del buf236
        buf239 = buf202; del buf202  # reuse
        buf240 = empty((112, ), device='cuda', dtype=torch.float32)
        buf241 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_57.run(buf198, buf237, convolution_49, unsqueeze_426, squeeze_88, buf239, buf240, buf241, 112, 1568, grid=grid(112), stream=stream0)
        buf242 = buf204; del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_58.run(buf198, buf237, convolution_49, unsqueeze_426, buf240, squeeze_88, buf239, primals_63, buf242, 896, 196, grid=grid(896, 196), stream=stream0)
        del convolution_49
        del primals_63
        del squeeze_88
        del unsqueeze_426
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf243 = aten.convolution_backward(buf242, mul_242, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf242
        del mul_242
        del primals_169
        buf244 = buf243[0]
        buf245 = buf243[1]
        del buf243
        buf246 = buf208; del buf208  # reuse
        # Source Nodes: [x_164], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_48.run(buf244, add_148, buf246, 10752, 98, grid=grid(10752), stream=stream0)
        buf247 = reinterpret_tensor(buf218, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf218  # reuse
        buf248 = reinterpret_tensor(buf247, (8, 672, 1, 1), (672, 1, 1, 1), 0); del buf247  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_164], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_49.run(buf248, buf246, convolution_48, 5376, 2, grid=grid(5376), stream=stream0)
        del buf246
        buf249 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_31.run(buf248, buf249, 672, 8, grid=grid(672), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf250 = aten.convolution_backward(buf248, mul_241, primals_167, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf248
        del mul_241
        del primals_167
        buf251 = buf250[0]
        buf252 = buf250[1]
        del buf250
        buf253 = buf251; del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_32.run(buf253, convolution_47, 224, grid=grid(224), stream=stream0)
        del convolution_47
        buf254 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_33.run(buf253, buf254, 28, 8, grid=grid(28), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf255 = aten.convolution_backward(buf253, mean_9, primals_165, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf253
        del mean_9
        del primals_165
        buf256 = buf255[0]
        buf257 = buf255[1]
        del buf255
        buf258 = reinterpret_tensor(buf232, (672, 13), (13, 1), 0); del buf232  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_50.run(buf244, convolution_48, buf256, add_148, buf258, 8736, 121, grid=grid(8736), stream=stream0)
        buf259 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_40.run(buf258, buf259, 672, 13, grid=grid(672), stream=stream0)
        buf260 = reinterpret_tensor(buf258, (672, 13), (1, 672), 0); del buf258  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_51.run(buf244, convolution_48, buf256, add_148, convolution_46, unsqueeze_438, buf260, 8736, 121, grid=grid(8736), stream=stream0)
        buf261 = empty((672, ), device='cuda', dtype=torch.float32)
        buf263 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_42.run(buf260, squeeze_85, buf261, buf263, 672, 13, grid=grid(672), stream=stream0)
        buf262 = reinterpret_tensor(buf235, (8, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf235  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_52.run(buf244, convolution_48, buf256, add_148, convolution_46, unsqueeze_438, buf261, squeeze_85, buf259, buf262, 1568, 672, grid=grid(1568, 672), stream=stream0)
        del add_148
        del buf256
        del convolution_46
        del convolution_48
        del unsqueeze_438
        buf264 = buf244; del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_53.run(buf262, squeeze_85, primals_61, buf264, 5376, 196, grid=grid(5376, 196), stream=stream0)
        del buf262
        del primals_61
        del squeeze_85
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf265 = aten.convolution_backward(buf264, mul_232, primals_164, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
        del buf264
        del mul_232
        del primals_164
        buf266 = buf265[0]
        buf267 = buf265[1]
        del buf265
        buf268 = reinterpret_tensor(buf260, (672, 13), (13, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_54.run(buf266, mul_689, buf268, 8736, 121, grid=grid(8736), stream=stream0)
        buf269 = buf261; del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_40.run(buf268, buf269, 672, 13, grid=grid(672), stream=stream0)
        buf270 = reinterpret_tensor(buf268, (672, 13), (1, 672), 0); del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_55.run(buf266, mul_689, convolution_45, unsqueeze_450, buf270, 8736, 121, grid=grid(8736), stream=stream0)
        buf271 = empty((672, ), device='cuda', dtype=torch.float32)
        buf272 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_42.run(buf270, squeeze_82, buf271, buf272, 672, 13, grid=grid(672), stream=stream0)
        del buf270
        buf273 = buf266; del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_56.run(buf273, mul_689, convolution_45, unsqueeze_450, buf271, squeeze_82, buf269, primals_59, 5376, 196, grid=grid(5376, 196), stream=stream0)
        del buf271
        del convolution_45
        del mul_689
        del primals_59
        del squeeze_82
        del unsqueeze_450
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf274 = aten.convolution_backward(buf273, add_138, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_138
        del buf273
        del primals_163
        buf275 = buf274[0]
        buf276 = buf274[1]
        del buf274
        buf277 = buf240; del buf240  # reuse
        buf278 = empty((112, ), device='cuda', dtype=torch.float32)
        buf280 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_59.run(buf198, buf237, buf275, convolution_44, unsqueeze_462, squeeze_79, buf277, buf278, buf280, 112, 1568, grid=grid(112), stream=stream0)
        buf279 = buf198; del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_60.run(buf279, buf237, buf275, convolution_44, unsqueeze_462, buf278, squeeze_79, buf277, primals_57, 896, 196, grid=grid(896, 196), stream=stream0)
        del buf237
        del buf275
        del buf278
        del convolution_44
        del primals_57
        del squeeze_79
        del unsqueeze_462
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf281 = aten.convolution_backward(buf279, mul_217, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf279
        del mul_217
        del primals_162
        buf282 = buf281[0]
        buf283 = buf281[1]
        del buf281
        buf284 = empty_strided((8, 480, 1, 1, 2), (960, 2, 7680, 7680, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_148], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_61.run(buf282, add_133, buf284, 7680, 98, grid=grid(7680), stream=stream0)
        buf285 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf286 = reinterpret_tensor(buf285, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf285  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_148], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_62.run(buf286, buf284, convolution_43, 3840, 2, grid=grid(3840), stream=stream0)
        buf287 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_63.run(buf286, buf287, 480, 8, grid=grid(480), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf288 = aten.convolution_backward(buf286, mul_216, primals_160, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf286
        del mul_216
        del primals_160
        buf289 = buf288[0]
        buf290 = buf288[1]
        del buf288
        buf291 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_64.run(buf291, convolution_42, 160, grid=grid(160), stream=stream0)
        del convolution_42
        buf292 = empty((20, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_65.run(buf291, buf292, 20, 8, grid=grid(20), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf293 = aten.convolution_backward(buf291, mean_8, primals_158, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf291
        del mean_8
        del primals_158
        buf294 = buf293[0]
        buf295 = buf293[1]
        del buf293
        buf296 = empty((480, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_66.run(buf282, convolution_43, buf294, add_133, buf296, 6240, 121, grid=grid(6240), stream=stream0)
        buf297 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67.run(buf296, buf297, 480, 13, grid=grid(480), stream=stream0)
        buf298 = reinterpret_tensor(buf296, (480, 13), (1, 480), 0); del buf296  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_68.run(buf282, convolution_43, buf294, add_133, convolution_41, unsqueeze_474, buf298, 6240, 121, grid=grid(6240), stream=stream0)
        buf299 = empty((480, ), device='cuda', dtype=torch.float32)
        buf301 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69.run(buf298, squeeze_76, buf299, buf301, 480, 13, grid=grid(480), stream=stream0)
        buf300 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70.run(buf282, convolution_43, buf294, add_133, convolution_41, unsqueeze_474, buf299, squeeze_76, buf297, buf300, 1568, 480, grid=grid(1568, 480), stream=stream0)
        del add_133
        del convolution_41
        del convolution_43
        del unsqueeze_474
        buf302 = buf282; del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_71.run(buf300, squeeze_76, primals_55, buf302, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf300
        del primals_55
        del squeeze_76
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf303 = aten.convolution_backward(buf302, mul_207, primals_157, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False])
        del buf302
        del mul_207
        del primals_157
        buf304 = buf303[0]
        buf305 = buf303[1]
        del buf303
        buf306 = reinterpret_tensor(buf298, (480, 13), (13, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_72.run(buf304, mul_729, buf306, 6240, 121, grid=grid(6240), stream=stream0)
        buf307 = buf299; del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67.run(buf306, buf307, 480, 13, grid=grid(480), stream=stream0)
        buf308 = reinterpret_tensor(buf306, (480, 13), (1, 480), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_73.run(buf304, mul_729, convolution_40, unsqueeze_486, buf308, 6240, 121, grid=grid(6240), stream=stream0)
        buf309 = empty((480, ), device='cuda', dtype=torch.float32)
        buf310 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69.run(buf308, squeeze_73, buf309, buf310, 480, 13, grid=grid(480), stream=stream0)
        buf311 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_74.run(buf311, mul_729, convolution_40, unsqueeze_486, buf309, squeeze_73, buf307, primals_53, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del convolution_40
        del mul_729
        del primals_53
        del squeeze_73
        del unsqueeze_486
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf312 = aten.convolution_backward(buf311, add_123, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_123
        del primals_156
        buf313 = buf312[0]
        buf314 = buf312[1]
        del buf312
        buf315 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_75.run(buf313, buf315, 80, 1568, grid=grid(80), stream=stream0)
        buf316 = empty((80, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_76.run(buf313, convolution_39, unsqueeze_498, buf316, 1040, 121, grid=grid(1040), stream=stream0)
        buf317 = empty((80, ), device='cuda', dtype=torch.float32)
        buf318 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_77.run(buf316, squeeze_70, buf317, buf318, 80, 13, grid=grid(80), stream=stream0)
        del buf316
        buf319 = reinterpret_tensor(buf16, (8, 80, 14, 14), (15680, 196, 14, 1), 0); del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_78.run(buf313, convolution_39, unsqueeze_498, buf317, squeeze_70, buf315, primals_51, buf319, 640, 196, grid=grid(640, 196), stream=stream0)
        del convolution_39
        del primals_51
        del squeeze_70
        del unsqueeze_498
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf320 = aten.convolution_backward(buf319, mul_192, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_192
        del primals_155
        buf321 = buf320[0]
        buf322 = buf320[1]
        del buf320
        buf323 = buf284; del buf284  # reuse
        # Source Nodes: [x_131], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_61.run(buf321, add_117, buf323, 7680, 98, grid=grid(7680), stream=stream0)
        buf324 = reinterpret_tensor(buf294, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf294  # reuse
        buf325 = reinterpret_tensor(buf324, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf324  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate, x_131], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_62.run(buf325, buf323, convolution_38, 3840, 2, grid=grid(3840), stream=stream0)
        buf326 = buf309; del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_63.run(buf325, buf326, 480, 8, grid=grid(480), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf327 = aten.convolution_backward(buf325, mul_191, primals_153, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf325
        del mul_191
        del primals_153
        buf328 = buf327[0]
        buf329 = buf327[1]
        del buf327
        buf330 = buf328; del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_64.run(buf330, convolution_37, 160, grid=grid(160), stream=stream0)
        del convolution_37
        buf331 = empty((20, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_65.run(buf330, buf331, 20, 8, grid=grid(20), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf332 = aten.convolution_backward(buf330, mean_7, primals_151, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf330
        del mean_7
        del primals_151
        buf333 = buf332[0]
        buf334 = buf332[1]
        del buf332
        buf335 = reinterpret_tensor(buf308, (480, 13), (13, 1), 0); del buf308  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_66.run(buf321, convolution_38, buf333, add_117, buf335, 6240, 121, grid=grid(6240), stream=stream0)
        buf336 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67.run(buf335, buf336, 480, 13, grid=grid(480), stream=stream0)
        buf337 = reinterpret_tensor(buf335, (480, 13), (1, 480), 0); del buf335  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_68.run(buf321, convolution_38, buf333, add_117, convolution_36, unsqueeze_510, buf337, 6240, 121, grid=grid(6240), stream=stream0)
        buf338 = empty((480, ), device='cuda', dtype=torch.float32)
        buf340 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69.run(buf337, squeeze_67, buf338, buf340, 480, 13, grid=grid(480), stream=stream0)
        buf339 = reinterpret_tensor(buf311, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf311  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70.run(buf321, convolution_38, buf333, add_117, convolution_36, unsqueeze_510, buf338, squeeze_67, buf336, buf339, 1568, 480, grid=grid(1568, 480), stream=stream0)
        del add_117
        del convolution_36
        del convolution_38
        del unsqueeze_510
        buf341 = buf321; del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_71.run(buf339, squeeze_67, primals_49, buf341, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf339
        del primals_49
        del squeeze_67
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf342 = aten.convolution_backward(buf341, mul_182, primals_150, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
        del buf341
        del mul_182
        del primals_150
        buf343 = buf342[0]
        buf344 = buf342[1]
        del buf342
        buf345 = reinterpret_tensor(buf337, (480, 13), (13, 1), 0); del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_72.run(buf343, mul_769, buf345, 6240, 121, grid=grid(6240), stream=stream0)
        buf346 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67.run(buf345, buf346, 480, 13, grid=grid(480), stream=stream0)
        buf347 = reinterpret_tensor(buf345, (480, 13), (1, 480), 0); del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_73.run(buf343, mul_769, convolution_35, unsqueeze_522, buf347, 6240, 121, grid=grid(6240), stream=stream0)
        buf348 = empty((480, ), device='cuda', dtype=torch.float32)
        buf349 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69.run(buf347, squeeze_64, buf348, buf349, 480, 13, grid=grid(480), stream=stream0)
        buf350 = buf343; del buf343  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_74.run(buf350, mul_769, convolution_35, unsqueeze_522, buf348, squeeze_64, buf346, primals_47, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del convolution_35
        del mul_769
        del primals_47
        del squeeze_64
        del unsqueeze_522
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf351 = aten.convolution_backward(buf350, add_107, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_107
        del primals_149
        buf352 = buf351[0]
        buf353 = buf351[1]
        del buf351
        buf354 = buf317; del buf317  # reuse
        buf355 = empty((80, ), device='cuda', dtype=torch.float32)
        buf356 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_79.run(buf313, buf352, convolution_34, unsqueeze_534, squeeze_61, buf354, buf355, buf356, 80, 1568, grid=grid(80), stream=stream0)
        buf357 = buf319; del buf319  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_80.run(buf313, buf352, convolution_34, unsqueeze_534, buf355, squeeze_61, buf354, primals_45, buf357, 640, 196, grid=grid(640, 196), stream=stream0)
        del convolution_34
        del primals_45
        del squeeze_61
        del unsqueeze_534
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf358 = aten.convolution_backward(buf357, mul_167, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf357
        del mul_167
        del primals_148
        buf359 = buf358[0]
        buf360 = buf358[1]
        del buf358
        buf361 = buf323; del buf323  # reuse
        # Source Nodes: [x_114], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_61.run(buf359, add_101, buf361, 7680, 98, grid=grid(7680), stream=stream0)
        buf362 = reinterpret_tensor(buf333, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf333  # reuse
        buf363 = reinterpret_tensor(buf362, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf362  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_114], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_62.run(buf363, buf361, convolution_33, 3840, 2, grid=grid(3840), stream=stream0)
        del buf361
        buf364 = buf348; del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_63.run(buf363, buf364, 480, 8, grid=grid(480), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf365 = aten.convolution_backward(buf363, mul_166, primals_146, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf363
        del mul_166
        del primals_146
        buf366 = buf365[0]
        buf367 = buf365[1]
        del buf365
        buf368 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_64.run(buf368, convolution_32, 160, grid=grid(160), stream=stream0)
        del convolution_32
        buf369 = empty((20, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_65.run(buf368, buf369, 20, 8, grid=grid(20), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf370 = aten.convolution_backward(buf368, mean_6, primals_144, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf368
        del mean_6
        del primals_144
        buf371 = buf370[0]
        buf372 = buf370[1]
        del buf370
        buf373 = reinterpret_tensor(buf347, (480, 13), (13, 1), 0); del buf347  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_66.run(buf359, convolution_33, buf371, add_101, buf373, 6240, 121, grid=grid(6240), stream=stream0)
        buf374 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67.run(buf373, buf374, 480, 13, grid=grid(480), stream=stream0)
        buf375 = reinterpret_tensor(buf373, (480, 13), (1, 480), 0); del buf373  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_68.run(buf359, convolution_33, buf371, add_101, convolution_31, unsqueeze_546, buf375, 6240, 121, grid=grid(6240), stream=stream0)
        buf376 = empty((480, ), device='cuda', dtype=torch.float32)
        buf378 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69.run(buf375, squeeze_58, buf376, buf378, 480, 13, grid=grid(480), stream=stream0)
        buf377 = reinterpret_tensor(buf350, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf350  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70.run(buf359, convolution_33, buf371, add_101, convolution_31, unsqueeze_546, buf376, squeeze_58, buf374, buf377, 1568, 480, grid=grid(1568, 480), stream=stream0)
        del add_101
        del convolution_31
        del convolution_33
        del unsqueeze_546
        buf379 = buf359; del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_71.run(buf377, squeeze_58, primals_43, buf379, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf377
        del primals_43
        del squeeze_58
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf380 = aten.convolution_backward(buf379, mul_157, primals_143, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
        del buf379
        del mul_157
        del primals_143
        buf381 = buf380[0]
        buf382 = buf380[1]
        del buf380
        buf383 = reinterpret_tensor(buf375, (480, 13), (13, 1), 0); del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_72.run(buf381, mul_809, buf383, 6240, 121, grid=grid(6240), stream=stream0)
        buf384 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67.run(buf383, buf384, 480, 13, grid=grid(480), stream=stream0)
        buf385 = reinterpret_tensor(buf383, (480, 13), (1, 480), 0); del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_73.run(buf381, mul_809, convolution_30, unsqueeze_558, buf385, 6240, 121, grid=grid(6240), stream=stream0)
        buf386 = empty((480, ), device='cuda', dtype=torch.float32)
        buf387 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69.run(buf385, squeeze_55, buf386, buf387, 480, 13, grid=grid(480), stream=stream0)
        del buf385
        buf388 = buf381; del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_74.run(buf388, mul_809, convolution_30, unsqueeze_558, buf386, squeeze_55, buf384, primals_41, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf386
        del convolution_30
        del mul_809
        del primals_41
        del squeeze_55
        del unsqueeze_558
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf389 = aten.convolution_backward(buf388, add_91, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_91
        del buf388
        del primals_142
        buf390 = buf389[0]
        buf391 = buf389[1]
        del buf389
        buf392 = buf355; del buf355  # reuse
        buf393 = empty((80, ), device='cuda', dtype=torch.float32)
        buf395 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_81.run(buf313, buf352, buf390, convolution_29, unsqueeze_570, squeeze_52, buf392, buf393, buf395, 80, 1568, grid=grid(80), stream=stream0)
        buf394 = buf313; del buf313  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_82.run(buf394, buf352, buf390, convolution_29, unsqueeze_570, buf393, squeeze_52, buf392, primals_39, 640, 196, grid=grid(640, 196), stream=stream0)
        del buf352
        del buf390
        del buf393
        del convolution_29
        del primals_39
        del squeeze_52
        del unsqueeze_570
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf396 = aten.convolution_backward(buf394, mul_142, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf394
        del mul_142
        del primals_141
        buf397 = buf396[0]
        buf398 = buf396[1]
        del buf396
        buf399 = reinterpret_tensor(buf371, (8, 240, 1, 1, 2), (480, 2, 3840, 3840, 1), 0); del buf371  # reuse
        # Source Nodes: [x_98], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_83.run(buf397, add_86, buf399, 3840, 98, grid=grid(3840), stream=stream0)
        buf400 = empty_strided((8, 240, 1, 1), (240, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf401 = reinterpret_tensor(buf400, (8, 240, 1, 1), (240, 1, 1, 1), 0); del buf400  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_98], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_84.run(buf401, buf399, convolution_28, 1920, 2, grid=grid(1920), stream=stream0)
        del buf399
        buf402 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_85.run(buf401, buf402, 240, 8, grid=grid(240), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf403 = aten.convolution_backward(buf401, mul_141, primals_139, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf401
        del mul_141
        del primals_139
        buf404 = buf403[0]
        buf405 = buf403[1]
        del buf403
        buf406 = buf404; del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_86.run(buf406, convolution_27, 80, grid=grid(80), stream=stream0)
        del convolution_27
        buf407 = empty((10, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_87.run(buf406, buf407, 10, 8, grid=grid(10), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf408 = aten.convolution_backward(buf406, mean_5, primals_137, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf406
        del mean_5
        del primals_137
        buf409 = buf408[0]
        buf410 = buf408[1]
        del buf408
        buf411 = empty((240, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_88.run(buf397, convolution_28, buf409, add_86, buf411, 3120, 121, grid=grid(3120), stream=stream0)
        buf412 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_89.run(buf411, buf412, 240, 13, grid=grid(240), stream=stream0)
        buf413 = reinterpret_tensor(buf411, (240, 13), (1, 240), 0); del buf411  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_90.run(buf397, convolution_28, buf409, add_86, convolution_26, unsqueeze_582, buf413, 3120, 121, grid=grid(3120), stream=stream0)
        buf414 = empty((240, ), device='cuda', dtype=torch.float32)
        buf416 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_91.run(buf413, squeeze_49, buf414, buf416, 240, 13, grid=grid(240), stream=stream0)
        del buf413
        buf415 = empty_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_92.run(buf397, convolution_28, buf409, add_86, convolution_26, unsqueeze_582, buf414, squeeze_49, buf412, buf415, 1568, 240, grid=grid(1568, 240), stream=stream0)
        del add_86
        del convolution_26
        del convolution_28
        del unsqueeze_582
        buf417 = buf397; del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_93.run(buf415, squeeze_49, primals_37, buf417, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del buf415
        del primals_37
        del squeeze_49
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf418 = aten.convolution_backward(buf417, constant_pad_nd_3, primals_36, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 240, [True, True, False])
        del buf417
        del constant_pad_nd_3
        del primals_36
        buf419 = buf418[0]
        buf420 = buf418[1]
        del buf418
        buf421 = empty((240, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_94.run(buf419, mul_849, buf421, 11760, 128, grid=grid(11760), stream=stream0)
        buf422 = buf414; del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_95.run(buf421, buf422, 240, 49, grid=grid(240), stream=stream0)
        buf423 = reinterpret_tensor(buf421, (240, 49), (1, 240), 0); del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_96.run(buf419, mul_849, convolution_25, unsqueeze_594, buf423, 11760, 128, grid=grid(11760), stream=stream0)
        buf424 = empty((240, ), device='cuda', dtype=torch.float32)
        buf425 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_97.run(buf423, squeeze_46, buf424, buf425, 240, 49, grid=grid(240), stream=stream0)
        buf426 = empty((8, 240, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_98.run(buf419, mul_849, convolution_25, unsqueeze_594, buf424, squeeze_46, buf422, primals_34, buf426, 1920, 784, grid=grid(1920, 784), stream=stream0)
        del buf419
        del convolution_25
        del mul_849
        del primals_34
        del squeeze_46
        del unsqueeze_594
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf427 = aten.convolution_backward(buf426, add_76, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_76
        del primals_136
        buf428 = buf427[0]
        buf429 = buf427[1]
        del buf427
        buf430 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_99.run(buf428, buf430, 40, 6272, grid=grid(40), stream=stream0)
        buf431 = empty((40, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_100.run(buf428, convolution_24, unsqueeze_606, buf431, 1960, 128, grid=grid(1960), stream=stream0)
        buf432 = empty((40, ), device='cuda', dtype=torch.float32)
        buf433 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_101.run(buf431, squeeze_43, buf432, buf433, 40, 49, grid=grid(40), stream=stream0)
        del buf431
        buf434 = empty((8, 40, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_102.run(buf428, convolution_24, unsqueeze_606, buf432, squeeze_43, buf430, primals_32, buf434, 320, 784, grid=grid(320, 784), stream=stream0)
        del convolution_24
        del primals_32
        del squeeze_43
        del unsqueeze_606
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf435 = aten.convolution_backward(buf434, mul_117, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf434
        del mul_117
        del primals_135
        buf436 = buf435[0]
        buf437 = buf435[1]
        del buf435
        buf438 = empty_strided((8, 240, 1, 1, 7), (1680, 7, 13440, 13440, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_79], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_103.run(buf436, add_70, buf438, 13440, 112, grid=grid(13440), stream=stream0)
        buf439 = reinterpret_tensor(buf409, (8, 240, 1, 1), (240, 1, 1920, 1920), 0); del buf409  # reuse
        buf440 = reinterpret_tensor(buf439, (8, 240, 1, 1), (240, 1, 1, 1), 0); del buf439  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_79], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_104.run(buf440, buf438, convolution_23, 1920, 7, grid=grid(1920), stream=stream0)
        del buf438
        buf441 = buf424; del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_85.run(buf440, buf441, 240, 8, grid=grid(240), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf442 = aten.convolution_backward(buf440, mul_116, primals_133, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf440
        del mul_116
        del primals_133
        buf443 = buf442[0]
        buf444 = buf442[1]
        del buf442
        buf445 = buf443; del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_86.run(buf445, convolution_22, 80, grid=grid(80), stream=stream0)
        del convolution_22
        buf446 = empty((10, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_87.run(buf445, buf446, 10, 8, grid=grid(10), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf447 = aten.convolution_backward(buf445, mean_4, primals_131, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf445
        del mean_4
        del primals_131
        buf448 = buf447[0]
        buf449 = buf447[1]
        del buf447
        buf450 = reinterpret_tensor(buf423, (240, 49), (49, 1), 0); del buf423  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_105.run(buf436, convolution_23, buf448, add_70, buf450, 11760, 128, grid=grid(11760), stream=stream0)
        buf451 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_95.run(buf450, buf451, 240, 49, grid=grid(240), stream=stream0)
        buf452 = reinterpret_tensor(buf450, (240, 49), (1, 240), 0); del buf450  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_106.run(buf436, convolution_23, buf448, add_70, convolution_21, unsqueeze_618, buf452, 11760, 128, grid=grid(11760), stream=stream0)
        buf453 = empty((240, ), device='cuda', dtype=torch.float32)
        buf455 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_97.run(buf452, squeeze_40, buf453, buf455, 240, 49, grid=grid(240), stream=stream0)
        buf454 = reinterpret_tensor(buf426, (8, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf426  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_107.run(buf436, convolution_23, buf448, add_70, convolution_21, unsqueeze_618, buf453, squeeze_40, buf451, buf454, 6272, 240, grid=grid(6272, 240), stream=stream0)
        del add_70
        del buf448
        del convolution_21
        del convolution_23
        del unsqueeze_618
        buf456 = buf436; del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_108.run(buf454, squeeze_40, primals_30, buf456, 1920, 784, grid=grid(1920, 784), stream=stream0)
        del buf454
        del primals_30
        del squeeze_40
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf457 = aten.convolution_backward(buf456, mul_107, primals_130, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False])
        del buf456
        del mul_107
        del primals_130
        buf458 = buf457[0]
        buf459 = buf457[1]
        del buf457
        buf460 = reinterpret_tensor(buf452, (240, 49), (49, 1), 0); del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_109.run(buf458, mul_889, buf460, 11760, 128, grid=grid(11760), stream=stream0)
        buf461 = buf453; del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_95.run(buf460, buf461, 240, 49, grid=grid(240), stream=stream0)
        buf462 = reinterpret_tensor(buf460, (240, 49), (1, 240), 0); del buf460  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_110.run(buf458, mul_889, convolution_20, unsqueeze_630, buf462, 11760, 128, grid=grid(11760), stream=stream0)
        buf463 = empty((240, ), device='cuda', dtype=torch.float32)
        buf464 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_97.run(buf462, squeeze_37, buf463, buf464, 240, 49, grid=grid(240), stream=stream0)
        del buf462
        buf465 = buf458; del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_111.run(buf465, mul_889, convolution_20, unsqueeze_630, buf463, squeeze_37, buf461, primals_28, 1920, 784, grid=grid(1920, 784), stream=stream0)
        del buf463
        del convolution_20
        del mul_889
        del primals_28
        del squeeze_37
        del unsqueeze_630
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf466 = aten.convolution_backward(buf465, add_60, primals_129, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_60
        del buf465
        del primals_129
        buf467 = buf466[0]
        buf468 = buf466[1]
        del buf466
        buf469 = buf432; del buf432  # reuse
        buf470 = empty((40, ), device='cuda', dtype=torch.float32)
        buf471 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_112.run(buf428, buf467, convolution_19, unsqueeze_642, squeeze_34, buf469, buf470, buf471, 40, 6272, grid=grid(40), stream=stream0)
        buf472 = buf428; del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_113.run(buf472, buf467, convolution_19, unsqueeze_642, buf470, squeeze_34, buf469, primals_26, 320, 784, grid=grid(320, 784), stream=stream0)
        del buf467
        del buf470
        del convolution_19
        del primals_26
        del squeeze_34
        del unsqueeze_642
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf473 = aten.convolution_backward(buf472, mul_92, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf472
        del mul_92
        del primals_128
        buf474 = buf473[0]
        buf475 = buf473[1]
        del buf473
        buf476 = empty_strided((8, 144, 1, 1, 7), (1008, 7, 8064, 8064, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_114.run(buf474, add_55, buf476, 8064, 112, grid=grid(8064), stream=stream0)
        buf477 = reinterpret_tensor(buf156, (8, 144, 1, 1), (144, 1, 1152, 1152), 0); del buf156  # reuse
        buf478 = reinterpret_tensor(buf477, (8, 144, 1, 1), (144, 1, 1, 1), 0); del buf477  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_63], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_115.run(buf478, buf476, convolution_18, 1152, 7, grid=grid(1152), stream=stream0)
        del buf476
        buf479 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_116.run(buf478, buf479, 144, 8, grid=grid(144), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf480 = aten.convolution_backward(buf478, mul_91, primals_126, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf478
        del mul_91
        del primals_126
        buf481 = buf480[0]
        buf482 = buf480[1]
        del buf480
        buf483 = buf481; del buf481  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_117.run(buf483, convolution_17, 48, grid=grid(48), stream=stream0)
        del convolution_17
        buf484 = empty((6, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_118.run(buf483, buf484, 6, 8, grid=grid(6), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf485 = aten.convolution_backward(buf483, mean_3, primals_124, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf483
        del mean_3
        del primals_124
        buf486 = buf485[0]
        buf487 = buf485[1]
        del buf485
        buf488 = empty((144, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_119.run(buf474, convolution_18, buf486, add_55, buf488, 7056, 128, grid=grid(7056), stream=stream0)
        buf489 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_120.run(buf488, buf489, 144, 49, grid=grid(144), stream=stream0)
        buf490 = reinterpret_tensor(buf488, (144, 49), (1, 144), 0); del buf488  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_121.run(buf474, convolution_18, buf486, add_55, convolution_16, unsqueeze_654, buf490, 7056, 128, grid=grid(7056), stream=stream0)
        buf491 = empty((144, ), device='cuda', dtype=torch.float32)
        buf493 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_122.run(buf490, squeeze_31, buf491, buf493, 144, 49, grid=grid(144), stream=stream0)
        del buf490
        buf492 = empty_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_123.run(buf474, convolution_18, buf486, add_55, convolution_16, unsqueeze_654, buf491, squeeze_31, buf489, buf492, 6272, 144, grid=grid(6272, 144), stream=stream0)
        del add_55
        del convolution_16
        del convolution_18
        del unsqueeze_654
        buf494 = buf474; del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_124.run(buf492, squeeze_31, primals_24, buf494, 1152, 784, grid=grid(1152, 784), stream=stream0)
        del buf492
        del primals_24
        del squeeze_31
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf495 = aten.convolution_backward(buf494, constant_pad_nd_2, primals_23, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 144, [True, True, False])
        del buf494
        del constant_pad_nd_2
        del primals_23
        buf496 = buf495[0]
        buf497 = buf495[1]
        del buf495
        buf498 = empty((144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_125.run(buf496, mul_929, buf498, 28224, 128, grid=grid(28224), stream=stream0)
        buf499 = buf491; del buf491  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_126.run(buf498, buf499, 144, 196, grid=grid(144), stream=stream0)
        buf500 = reinterpret_tensor(buf498, (144, 196), (1, 144), 0); del buf498  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_127.run(buf496, mul_929, convolution_15, unsqueeze_666, buf500, 28224, 128, grid=grid(28224), stream=stream0)
        buf501 = empty((144, ), device='cuda', dtype=torch.float32)
        buf502 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_128.run(buf500, squeeze_28, buf501, buf502, 144, 196, grid=grid(144), stream=stream0)
        buf503 = empty((8, 144, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_129.run(buf496, mul_929, convolution_15, unsqueeze_666, buf501, squeeze_28, buf499, primals_21, buf503, 1152, 3136, grid=grid(1152, 3136), stream=stream0)
        del buf496
        del convolution_15
        del mul_929
        del primals_21
        del squeeze_28
        del unsqueeze_666
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf504 = aten.convolution_backward(buf503, add_45, primals_123, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_45
        del primals_123
        buf505 = buf504[0]
        buf506 = buf504[1]
        del buf504
        buf507 = empty_strided((24, 4), (1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_130.run(buf505, buf507, 96, 6272, grid=grid(96), stream=stream0)
        buf508 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_131.run(buf507, buf508, 24, 4, grid=grid(24), stream=stream0)
        buf509 = empty((24, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_132.run(buf505, convolution_14, unsqueeze_678, buf509, 4704, 128, grid=grid(4704), stream=stream0)
        buf510 = empty((24, ), device='cuda', dtype=torch.float32)
        buf511 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_133.run(buf509, squeeze_25, buf510, buf511, 24, 196, grid=grid(24), stream=stream0)
        del buf509
        buf512 = empty((8, 24, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_134.run(buf505, convolution_14, unsqueeze_678, buf510, squeeze_25, buf508, primals_19, buf512, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del convolution_14
        del primals_19
        del squeeze_25
        del unsqueeze_678
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf513 = aten.convolution_backward(buf512, mul_67, primals_122, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf512
        del mul_67
        del primals_122
        buf514 = buf513[0]
        buf515 = buf513[1]
        del buf513
        buf516 = empty_strided((8, 144, 1, 1, 25), (3600, 25, 28800, 28800, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_135.run(buf514, add_39, buf516, 28800, 126, grid=grid(28800), stream=stream0)
        buf517 = reinterpret_tensor(buf486, (8, 144, 1, 1), (144, 1, 1152, 1152), 0); del buf486  # reuse
        buf518 = reinterpret_tensor(buf517, (8, 144, 1, 1), (144, 1, 1, 1), 0); del buf517  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_44], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_136.run(buf518, buf516, convolution_13, 1152, 25, grid=grid(1152), stream=stream0)
        del buf516
        buf519 = buf501; del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_116.run(buf518, buf519, 144, 8, grid=grid(144), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf520 = aten.convolution_backward(buf518, mul_66, primals_120, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf518
        del mul_66
        del primals_120
        buf521 = buf520[0]
        buf522 = buf520[1]
        del buf520
        buf523 = buf521; del buf521  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_117.run(buf523, convolution_12, 48, grid=grid(48), stream=stream0)
        del convolution_12
        buf524 = empty((6, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_118.run(buf523, buf524, 6, 8, grid=grid(6), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf525 = aten.convolution_backward(buf523, mean_2, primals_118, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf523
        del mean_2
        del primals_118
        buf526 = buf525[0]
        buf527 = buf525[1]
        del buf525
        buf528 = reinterpret_tensor(buf500, (144, 196), (196, 1), 0); del buf500  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_137.run(buf514, convolution_13, buf526, add_39, buf528, 28224, 128, grid=grid(28224), stream=stream0)
        buf529 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_126.run(buf528, buf529, 144, 196, grid=grid(144), stream=stream0)
        buf530 = reinterpret_tensor(buf528, (144, 196), (1, 144), 0); del buf528  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_138.run(buf514, convolution_13, buf526, add_39, convolution_11, unsqueeze_690, buf530, 28224, 128, grid=grid(28224), stream=stream0)
        buf531 = empty((144, ), device='cuda', dtype=torch.float32)
        buf533 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_128.run(buf530, squeeze_22, buf531, buf533, 144, 196, grid=grid(144), stream=stream0)
        buf532 = reinterpret_tensor(buf503, (8, 144, 56, 56), (451584, 1, 8064, 144), 0); del buf503  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_139.run(buf514, convolution_13, buf526, add_39, convolution_11, unsqueeze_690, buf531, squeeze_22, buf529, buf532, 25088, 144, grid=grid(25088, 144), stream=stream0)
        del add_39
        del buf526
        del convolution_11
        del convolution_13
        del unsqueeze_690
        buf534 = buf514; del buf514  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_140.run(buf532, squeeze_22, primals_17, buf534, 1152, 3136, grid=grid(1152, 3136), stream=stream0)
        del buf532
        del primals_17
        del squeeze_22
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf535 = aten.convolution_backward(buf534, mul_57, primals_117, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False])
        del buf534
        del mul_57
        del primals_117
        buf536 = buf535[0]
        buf537 = buf535[1]
        del buf535
        buf538 = reinterpret_tensor(buf530, (144, 196), (196, 1), 0); del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_141.run(buf536, mul_969, buf538, 28224, 128, grid=grid(28224), stream=stream0)
        buf539 = buf531; del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_126.run(buf538, buf539, 144, 196, grid=grid(144), stream=stream0)
        buf540 = reinterpret_tensor(buf538, (144, 196), (1, 144), 0); del buf538  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_142.run(buf536, mul_969, convolution_10, unsqueeze_702, buf540, 28224, 128, grid=grid(28224), stream=stream0)
        buf541 = empty((144, ), device='cuda', dtype=torch.float32)
        buf542 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_128.run(buf540, squeeze_19, buf541, buf542, 144, 196, grid=grid(144), stream=stream0)
        del buf540
        buf543 = buf536; del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_143.run(buf543, mul_969, convolution_10, unsqueeze_702, buf541, squeeze_19, buf539, primals_15, 1152, 3136, grid=grid(1152, 3136), stream=stream0)
        del buf541
        del convolution_10
        del mul_969
        del primals_15
        del squeeze_19
        del unsqueeze_702
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf544 = aten.convolution_backward(buf543, add_29, primals_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_29
        del buf543
        del primals_116
        buf545 = buf544[0]
        buf546 = buf544[1]
        del buf544
        buf547 = buf507; del buf507  # reuse
        buf549 = empty_strided((24, 4), (1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_144.run(buf505, buf545, convolution_9, unsqueeze_714, buf547, buf549, 96, 6272, grid=grid(96), stream=stream0)
        buf548 = buf510; del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_131.run(buf547, buf548, 24, 4, grid=grid(24), stream=stream0)
        buf550 = empty((24, ), device='cuda', dtype=torch.float32)
        buf551 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_145.run(buf549, squeeze_16, buf550, buf551, 24, 4, grid=grid(24), stream=stream0)
        buf552 = buf505; del buf505  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_146.run(buf552, buf545, convolution_9, unsqueeze_714, buf550, squeeze_16, buf548, primals_13, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del buf545
        del buf550
        del convolution_9
        del primals_13
        del squeeze_16
        del unsqueeze_714
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf553 = aten.convolution_backward(buf552, mul_42, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf552
        del mul_42
        del primals_115
        buf554 = buf553[0]
        buf555 = buf553[1]
        del buf553
        buf556 = empty_strided((8, 96, 1, 1, 25), (2400, 25, 19200, 19200, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_147.run(buf554, add_24, buf556, 19200, 126, grid=grid(19200), stream=stream0)
        buf557 = reinterpret_tensor(buf51, (8, 96, 1, 1), (96, 1, 768, 768), 0); del buf51  # reuse
        buf558 = reinterpret_tensor(buf557, (8, 96, 1, 1), (96, 1, 1, 1), 0); del buf557  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_28], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_148.run(buf558, buf556, convolution_8, 768, 25, grid=grid(768), stream=stream0)
        del buf556
        buf559 = reinterpret_tensor(buf549, (96, ), (1, ), 0); del buf549  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_149.run(buf558, buf559, 96, 8, grid=grid(96), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf560 = aten.convolution_backward(buf558, mul_41, primals_113, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf558
        del mul_41
        del primals_113
        buf561 = buf560[0]
        buf562 = buf560[1]
        del buf560
        buf563 = buf561; del buf561  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_150.run(buf563, convolution_7, 32, grid=grid(32), stream=stream0)
        del convolution_7
        buf564 = empty((4, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_151.run(buf563, buf564, 4, 8, grid=grid(4), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf565 = aten.convolution_backward(buf563, mean_1, primals_111, [4], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_1
        del primals_111
        buf566 = buf565[0]
        buf567 = buf565[1]
        del buf565
        buf568 = empty((96, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_152.run(buf554, convolution_8, buf566, add_24, buf568, 18816, 128, grid=grid(18816), stream=stream0)
        buf569 = reinterpret_tensor(buf547, (96, ), (1, ), 0); del buf547  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_153.run(buf568, buf569, 96, 196, grid=grid(96), stream=stream0)
        buf570 = reinterpret_tensor(buf568, (96, 196), (1, 96), 0); del buf568  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_154.run(buf554, convolution_8, buf566, add_24, convolution_6, unsqueeze_726, buf570, 18816, 128, grid=grid(18816), stream=stream0)
        buf571 = empty((96, ), device='cuda', dtype=torch.float32)
        buf573 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_155.run(buf570, squeeze_13, buf571, buf573, 96, 196, grid=grid(96), stream=stream0)
        del buf570
        buf572 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_156.run(buf554, convolution_8, buf566, add_24, convolution_6, unsqueeze_726, buf571, squeeze_13, buf569, buf572, 25088, 96, grid=grid(25088, 96), stream=stream0)
        del add_24
        del buf566
        del convolution_6
        del convolution_8
        del unsqueeze_726
        buf574 = buf554; del buf554  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_157.run(buf572, squeeze_13, primals_11, buf574, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del buf572
        del primals_11
        del squeeze_13
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf575 = aten.convolution_backward(buf574, constant_pad_nd_1, primals_10, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 96, [True, True, False])
        del buf574
        del constant_pad_nd_1
        del primals_10
        buf576 = buf575[0]
        buf577 = buf575[1]
        del buf575
        buf578 = reinterpret_tensor(buf166, (96, 784), (784, 1), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_158.run(buf576, mul_1009, buf578, 75264, 128, grid=grid(75264), stream=stream0)
        buf579 = buf571; del buf571  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_constant_pad_nd_mul_native_batch_norm_backward_159.run(buf578, buf579, 96, 784, grid=grid(96), stream=stream0)
        buf580 = reinterpret_tensor(buf578, (96, 784), (1, 96), 0); del buf578  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_160.run(buf576, mul_1009, convolution_5, unsqueeze_738, buf580, 75264, 128, grid=grid(75264), stream=stream0)
        buf581 = empty((96, ), device='cuda', dtype=torch.float32)
        buf582 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_constant_pad_nd_mul_native_batch_norm_backward_161.run(buf580, squeeze_10, buf581, buf582, 96, 784, grid=grid(96), stream=stream0)
        del buf580
        buf583 = empty((8, 96, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_162.run(buf576, mul_1009, convolution_5, unsqueeze_738, buf581, squeeze_10, buf579, primals_8, buf583, 768, 12544, grid=grid(768, 12544), stream=stream0)
        del buf576
        del buf581
        del convolution_5
        del mul_1009
        del primals_8
        del squeeze_10
        del unsqueeze_738
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf584 = aten.convolution_backward(buf583, add_14, primals_110, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_14
        del buf583
        del primals_110
        buf585 = buf584[0]
        buf586 = buf584[1]
        del buf584
        buf587 = empty((16, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_163.run(buf585, buf587, 208, 7720, grid=grid(208), stream=stream0)
        buf588 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_164.run(buf587, buf588, 16, 13, grid=grid(16), stream=stream0)
        del buf587
        buf589 = empty((16, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_165.run(buf585, convolution_4, unsqueeze_750, buf589, 12544, 128, grid=grid(12544), stream=stream0)
        buf590 = empty((16, ), device='cuda', dtype=torch.float32)
        buf591 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_166.run(buf589, squeeze_7, buf590, buf591, 16, 784, grid=grid(16), stream=stream0)
        del buf589
        buf592 = buf585; del buf585  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_167.run(buf592, convolution_4, unsqueeze_750, buf590, squeeze_7, buf588, primals_6, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del buf590
        del convolution_4
        del primals_6
        del squeeze_7
        del unsqueeze_750
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf593 = aten.convolution_backward(buf592, mul_17, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf592
        del mul_17
        del primals_109
        buf594 = buf593[0]
        buf595 = buf593[1]
        del buf593
        buf596 = empty_strided((8, 32, 1, 1, 98), (3136, 98, 25088, 25088, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_168.run(buf594, add_9, buf596, 25088, 128, grid=grid(25088), stream=stream0)
        buf597 = empty_strided((8, 32, 1, 1), (32, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf598 = reinterpret_tensor(buf597, (8, 32, 1, 1), (32, 1, 1, 1), 0); del buf597  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_10], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_169.run(buf598, buf596, convolution_3, 256, 98, grid=grid(256), stream=stream0)
        buf599 = reinterpret_tensor(buf563, (32, ), (1, ), 0); del buf563  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_170.run(buf598, buf599, 32, 8, grid=grid(32), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf600 = aten.convolution_backward(buf598, mul_16, primals_107, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf598
        del mul_16
        del primals_107
        buf601 = buf600[0]
        buf602 = buf600[1]
        del buf600
        buf603 = buf601; del buf601  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_171.run(buf603, convolution_2, 64, grid=grid(64), stream=stream0)
        del convolution_2
        buf604 = empty((8, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_172.run(buf603, buf604, 8, 8, grid=grid(8), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf605 = aten.convolution_backward(buf603, mean, primals_105, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf603
        del mean
        del primals_105
        buf606 = buf605[0]
        buf607 = buf605[1]
        del buf605
        buf608 = reinterpret_tensor(buf596, (32, 784), (784, 1), 0); del buf596  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_173.run(buf594, convolution_3, buf606, add_9, buf608, 25088, 128, grid=grid(25088), stream=stream0)
        buf609 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_174.run(buf608, buf609, 32, 784, grid=grid(32), stream=stream0)
        buf610 = reinterpret_tensor(buf608, (32, 784), (1, 32), 0); del buf608  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_175.run(buf594, convolution_3, buf606, add_9, convolution_1, unsqueeze_762, buf610, 25088, 128, grid=grid(25088), stream=stream0)
        buf611 = empty((32, ), device='cuda', dtype=torch.float32)
        buf613 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_176.run(buf610, squeeze_4, buf611, buf613, 32, 784, grid=grid(32), stream=stream0)
        buf612 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_177.run(buf594, convolution_3, buf606, add_9, convolution_1, unsqueeze_762, buf611, squeeze_4, buf609, buf612, 100352, 32, grid=grid(100352, 32), stream=stream0)
        del add_9
        del buf606
        del convolution_1
        del convolution_3
        del unsqueeze_762
        buf614 = buf594; del buf594  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_178.run(buf612, squeeze_4, primals_4, buf614, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del buf612
        del primals_4
        del squeeze_4
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf615 = aten.convolution_backward(buf614, mul_7, primals_104, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf614
        del mul_7
        del primals_104
        buf616 = buf615[0]
        buf617 = buf615[1]
        del buf615
        buf618 = reinterpret_tensor(buf610, (32, 784), (784, 1), 0); del buf610  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_179.run(buf616, mul_1049, buf618, 25088, 128, grid=grid(25088), stream=stream0)
        buf619 = buf611; del buf611  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_174.run(buf618, buf619, 32, 784, grid=grid(32), stream=stream0)
        buf620 = reinterpret_tensor(buf618, (32, 784), (1, 32), 0); del buf618  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_180.run(buf616, mul_1049, convolution, unsqueeze_774, buf620, 25088, 128, grid=grid(25088), stream=stream0)
        buf621 = empty((32, ), device='cuda', dtype=torch.float32)
        buf622 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_176.run(buf620, squeeze_1, buf621, buf622, 32, 784, grid=grid(32), stream=stream0)
        del buf620
        buf623 = buf616; del buf616  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_181.run(buf623, mul_1049, convolution, unsqueeze_774, buf621, squeeze_1, buf619, primals_2, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del buf621
        del convolution
        del mul_1049
        del primals_2
        del squeeze_1
        del unsqueeze_774
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf624 = aten.convolution_backward(buf623, constant_pad_nd, primals_1, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf623
        del constant_pad_nd
        del primals_1
        buf625 = buf624[1]
        return (buf625, buf622, buf619, buf613, buf609, buf591, buf588, buf582, buf579, buf577, buf573, buf569, buf551, buf548, buf542, buf539, buf533, buf529, buf511, buf508, buf502, buf499, buf497, buf493, buf489, buf471, buf469, buf464, buf461, buf455, buf451, buf433, buf430, buf425, buf422, buf420, buf416, buf412, buf395, buf392, buf387, buf384, buf378, buf374, buf356, buf354, buf349, buf346, buf340, buf336, buf318, buf315, buf310, buf307, buf301, buf297, buf280, buf277, buf272, buf269, buf263, buf259, buf241, buf239, buf234, buf231, buf225, buf221, buf203, buf200, buf195, buf192, buf190, buf186, buf182, buf165, buf162, buf157, buf154, buf148, buf144, buf128, buf125, buf120, buf117, buf111, buf107, buf90, buf88, buf83, buf80, buf74, buf70, buf53, buf50, buf45, buf42, buf36, buf32, buf15, buf12, buf7, buf4, buf617, buf607, buf604, buf602, buf599, buf595, buf586, buf567, buf564, buf562, buf559, buf555, buf546, buf537, buf527, buf524, buf522, buf519, buf515, buf506, buf487, buf484, buf482, buf479, buf475, buf468, buf459, buf449, buf446, buf444, buf441, buf437, buf429, buf410, buf407, buf405, buf402, buf398, buf391, buf382, buf372, buf369, buf367, buf364, buf360, buf353, buf344, buf334, buf331, buf329, buf326, buf322, buf314, buf305, buf295, buf292, buf290, buf287, buf283, buf276, buf267, buf257, buf254, buf252, buf249, buf245, buf238, buf229, buf219, buf216, buf214, buf211, buf207, buf199, buf180, buf177, buf175, buf172, buf169, buf161, buf152, buf142, buf139, buf137, buf134, buf131, buf124, buf115, buf105, buf102, buf100, buf97, buf94, buf87, buf78, buf68, buf65, buf63, buf60, buf57, buf49, buf40, buf30, buf27, buf25, buf22, buf19, buf11, reinterpret_tensor(buf1, (1000, 1280), (1280, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((4, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((96, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((40, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((192, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    constant_pad_nd = rand_strided((8, 3, 225, 225), (151875, 1, 675, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_7 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_9 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cuda:0', dtype=torch.float32)
    mul_16 = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    mul_17 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_14 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_1 = rand_strided((8, 96, 113, 113), (1225824, 1, 10848, 96), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_24 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    mean_1 = rand_strided((8, 96, 1, 1), (96, 1, 96, 96), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 4, 1, 1), (4, 1, 4, 4), device='cuda:0', dtype=torch.float32)
    mul_41 = rand_strided((8, 4, 1, 1), (4, 1, 4, 4), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 96, 1, 1), (96, 1, 96, 96), device='cuda:0', dtype=torch.float32)
    mul_42 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_29 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_57 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_39 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    mean_2 = rand_strided((8, 144, 1, 1), (144, 1, 144, 144), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 6, 1, 1), (6, 1, 6, 6), device='cuda:0', dtype=torch.float32)
    mul_66 = rand_strided((8, 6, 1, 1), (6, 1, 6, 6), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 144, 1, 1), (144, 1, 144, 144), device='cuda:0', dtype=torch.float32)
    mul_67 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_45 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_2 = rand_strided((8, 144, 59, 59), (501264, 1, 8496, 144), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_55 = rand_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cuda:0', dtype=torch.float32)
    mean_3 = rand_strided((8, 144, 1, 1), (144, 1, 144, 144), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 6, 1, 1), (6, 1, 6, 6), device='cuda:0', dtype=torch.float32)
    mul_91 = rand_strided((8, 6, 1, 1), (6, 1, 6, 6), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 144, 1, 1), (144, 1, 144, 144), device='cuda:0', dtype=torch.float32)
    mul_92 = rand_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_60 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_107 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_70 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    mean_4 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 10, 1, 1), (10, 1, 10, 10), device='cuda:0', dtype=torch.float32)
    mul_116 = rand_strided((8, 10, 1, 1), (10, 1, 10, 10), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cuda:0', dtype=torch.float32)
    mul_117 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_76 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_3 = rand_strided((8, 240, 29, 29), (201840, 1, 6960, 240), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_86 = rand_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cuda:0', dtype=torch.float32)
    mean_5 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 10, 1, 1), (10, 1, 10, 10), device='cuda:0', dtype=torch.float32)
    mul_141 = rand_strided((8, 10, 1, 1), (10, 1, 10, 10), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cuda:0', dtype=torch.float32)
    mul_142 = rand_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_91 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_157 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_101 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    mean_6 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda:0', dtype=torch.float32)
    mul_166 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    mul_167 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_107 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_182 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_117 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    mean_7 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda:0', dtype=torch.float32)
    mul_191 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    mul_192 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_123 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_207 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_133 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    mean_8 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda:0', dtype=torch.float32)
    mul_216 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    mul_217 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_138 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_232 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_148 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    mean_9 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    mul_241 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    mul_242 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_154 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_257 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_164 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    mean_10 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    mul_266 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    mul_267 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_170 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_4 = rand_strided((8, 672, 17, 17), (194208, 1, 11424, 672), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_180 = rand_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda:0', dtype=torch.float32)
    mean_11 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    convolution_57 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    mul_291 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    convolution_58 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    mul_292 = rand_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda:0', dtype=torch.float32)
    convolution_59 = rand_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_185 = rand_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_307 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_195 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    mean_12 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    convolution_62 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    mul_316 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    convolution_63 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    mul_317 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    convolution_64 = rand_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_201 = rand_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    convolution_65 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_332 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    convolution_66 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_211 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    mean_13 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    convolution_67 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    mul_341 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    convolution_68 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    mul_342 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    convolution_69 = rand_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_217 = rand_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    convolution_70 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_357 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    convolution_71 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_227 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    mean_14 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    convolution_72 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    mul_366 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    convolution_73 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    mul_367 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    convolution_74 = rand_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_233 = rand_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    convolution_75 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_382 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    convolution_76 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_243 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    mean_15 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    convolution_77 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    mul_391 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    convolution_78 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    mul_392 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    convolution_79 = rand_strided((8, 320, 7, 7), (15680, 1, 2240, 320), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_248 = rand_strided((8, 320, 7, 7), (15680, 1, 2240, 320), device='cuda:0', dtype=torch.float32)
    convolution_80 = rand_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((8, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_409 = rand_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda:0', dtype=torch.float32)
    unsqueeze_198 = rand_strided((1, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_210 = rand_strided((1, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_222 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_449 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    unsqueeze_234 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_246 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_489 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    unsqueeze_270 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_282 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_294 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_529 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    unsqueeze_306 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_318 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_569 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    unsqueeze_342 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_354 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_366 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_609 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    unsqueeze_378 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_390 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_649 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    unsqueeze_414 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_426 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_438 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_689 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    unsqueeze_450 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_462 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_729 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    unsqueeze_486 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_498 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_510 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_769 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    unsqueeze_522 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_534 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_546 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_809 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    unsqueeze_558 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_570 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_582 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_849 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    unsqueeze_594 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_606 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_618 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_889 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    unsqueeze_630 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_642 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_654 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_929 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    unsqueeze_666 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_678 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_690 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_969 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    unsqueeze_702 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_714 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_726 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_1009 = rand_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda:0', dtype=torch.float32)
    unsqueeze_738 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_750 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_762 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_1049 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    unsqueeze_774 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_6, primals_8, primals_10, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_105, primals_107, primals_109, primals_110, primals_111, primals_113, primals_115, primals_116, primals_117, primals_118, primals_120, primals_122, primals_123, primals_124, primals_126, primals_128, primals_129, primals_130, primals_131, primals_133, primals_135, primals_136, primals_137, primals_139, primals_141, primals_142, primals_143, primals_144, primals_146, primals_148, primals_149, primals_150, primals_151, primals_153, primals_155, primals_156, primals_157, primals_158, primals_160, primals_162, primals_163, primals_164, primals_165, primals_167, primals_169, primals_170, primals_171, primals_172, primals_174, primals_176, primals_177, primals_178, primals_180, primals_182, primals_183, primals_184, primals_185, primals_187, primals_189, primals_190, primals_191, primals_192, primals_194, primals_196, primals_197, primals_198, primals_199, primals_201, primals_203, primals_204, primals_205, primals_206, primals_208, primals_210, primals_211, constant_pad_nd, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, add_9, mean, convolution_2, mul_16, convolution_3, mul_17, convolution_4, squeeze_7, add_14, convolution_5, squeeze_10, constant_pad_nd_1, convolution_6, squeeze_13, add_24, mean_1, convolution_7, mul_41, convolution_8, mul_42, convolution_9, squeeze_16, add_29, convolution_10, squeeze_19, mul_57, convolution_11, squeeze_22, add_39, mean_2, convolution_12, mul_66, convolution_13, mul_67, convolution_14, squeeze_25, add_45, convolution_15, squeeze_28, constant_pad_nd_2, convolution_16, squeeze_31, add_55, mean_3, convolution_17, mul_91, convolution_18, mul_92, convolution_19, squeeze_34, add_60, convolution_20, squeeze_37, mul_107, convolution_21, squeeze_40, add_70, mean_4, convolution_22, mul_116, convolution_23, mul_117, convolution_24, squeeze_43, add_76, convolution_25, squeeze_46, constant_pad_nd_3, convolution_26, squeeze_49, add_86, mean_5, convolution_27, mul_141, convolution_28, mul_142, convolution_29, squeeze_52, add_91, convolution_30, squeeze_55, mul_157, convolution_31, squeeze_58, add_101, mean_6, convolution_32, mul_166, convolution_33, mul_167, convolution_34, squeeze_61, add_107, convolution_35, squeeze_64, mul_182, convolution_36, squeeze_67, add_117, mean_7, convolution_37, mul_191, convolution_38, mul_192, convolution_39, squeeze_70, add_123, convolution_40, squeeze_73, mul_207, convolution_41, squeeze_76, add_133, mean_8, convolution_42, mul_216, convolution_43, mul_217, convolution_44, squeeze_79, add_138, convolution_45, squeeze_82, mul_232, convolution_46, squeeze_85, add_148, mean_9, convolution_47, mul_241, convolution_48, mul_242, convolution_49, squeeze_88, add_154, convolution_50, squeeze_91, mul_257, convolution_51, squeeze_94, add_164, mean_10, convolution_52, mul_266, convolution_53, mul_267, convolution_54, squeeze_97, add_170, convolution_55, squeeze_100, constant_pad_nd_4, convolution_56, squeeze_103, add_180, mean_11, convolution_57, mul_291, convolution_58, mul_292, convolution_59, squeeze_106, add_185, convolution_60, squeeze_109, mul_307, convolution_61, squeeze_112, add_195, mean_12, convolution_62, mul_316, convolution_63, mul_317, convolution_64, squeeze_115, add_201, convolution_65, squeeze_118, mul_332, convolution_66, squeeze_121, add_211, mean_13, convolution_67, mul_341, convolution_68, mul_342, convolution_69, squeeze_124, add_217, convolution_70, squeeze_127, mul_357, convolution_71, squeeze_130, add_227, mean_14, convolution_72, mul_366, convolution_73, mul_367, convolution_74, squeeze_133, add_233, convolution_75, squeeze_136, mul_382, convolution_76, squeeze_139, add_243, mean_15, convolution_77, mul_391, convolution_78, mul_392, convolution_79, squeeze_142, add_248, convolution_80, squeeze_145, view, permute_1, mul_409, unsqueeze_198, unsqueeze_210, unsqueeze_222, mul_449, unsqueeze_234, unsqueeze_246, unsqueeze_258, mul_489, unsqueeze_270, unsqueeze_282, unsqueeze_294, mul_529, unsqueeze_306, unsqueeze_318, unsqueeze_330, mul_569, unsqueeze_342, unsqueeze_354, unsqueeze_366, mul_609, unsqueeze_378, unsqueeze_390, unsqueeze_402, mul_649, unsqueeze_414, unsqueeze_426, unsqueeze_438, mul_689, unsqueeze_450, unsqueeze_462, unsqueeze_474, mul_729, unsqueeze_486, unsqueeze_498, unsqueeze_510, mul_769, unsqueeze_522, unsqueeze_534, unsqueeze_546, mul_809, unsqueeze_558, unsqueeze_570, unsqueeze_582, mul_849, unsqueeze_594, unsqueeze_606, unsqueeze_618, mul_889, unsqueeze_630, unsqueeze_642, unsqueeze_654, mul_929, unsqueeze_666, unsqueeze_678, unsqueeze_690, mul_969, unsqueeze_702, unsqueeze_714, unsqueeze_726, mul_1009, unsqueeze_738, unsqueeze_750, unsqueeze_762, mul_1049, unsqueeze_774, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tf_efficientnet_b0', benchmark_compiled_module)
