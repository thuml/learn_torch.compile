
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


# kernel path: /tmp/torchinductor_youkaichao/5x/c5xqbgrmeg42dkysjggb6z5lqjowrn5zpzvsc7km2gln565cxor5.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (27*y1)), tmp0, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/qa/cqalvmmjjwukju4qqja3d7v6rdgzeqejmbflio2hppxk5qiphv6g.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 73728
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (1728*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2i/c2iqmtfdd63vbzgqb6pdqxjnhw6gal4yp7brb2fedxhycadqmcyy.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 294912
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (384*x2) + (3456*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j6/cj6sxxaqkda222mhhtggl37fb4ue65g4pvnggnbktwq6zv7i2knc.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (50176*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (150528*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v6/cv64p6dvv42bg2fcivq64472tdrvujdoyymzlid2lm5casiex63j.py
# Source Nodes: [l__mod___patch_embed_proj_0_0], Original ATen: [aten.convolution]
# l__mod___patch_embed_proj_0_0 => convolution
triton_poi_fused_convolution_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 12544
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
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (2408448*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/so/csohcb3ydaqs6kxrqp42ec533ezei3d7hr3dkwopja5yqqciy4u5.py
# Source Nodes: [l__mod___patch_embed_proj_0_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___patch_embed_proj_0_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (37632*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xq/cxqehljzj3fw5fdeb3w5ren63aaete7zslr2mlgtejdqvplminks.py
# Source Nodes: [l__mod___patch_embed_proj_0_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___patch_embed_proj_0_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (192*r2) + (24576*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (192*r2) + (24576*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (192*r2) + (24576*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (192*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (192*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (192*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3t/c3trxx6a5pdx3ww2lndrgklwtnikgwynna4ofgtyna2rr63wk32m.py
# Source Nodes: [l__mod___patch_embed_proj_0_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___patch_embed_proj_0_1 => add_1, add_2, add_3, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze_1, var_mean
triton_per_fused__native_batch_norm_legit_functional_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_7', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 100352.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.00000996502277
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ow/cowti3nfph5wqddmjxfyuo4twbghfmgmiarcvn7wrat73p35ng2t.py
# Source Nodes: [l__mod___patch_embed_proj_0_1, l__mod___patch_embed_proj_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu, aten.gelu_backward]
# l__mod___patch_embed_proj_0_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
# l__mod___patch_embed_proj_1 => add_5, erf, mul_7, mul_8, mul_9
triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19267584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 100352.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.5
    tmp15 = tmp13 * tmp14
    tmp16 = 0.7071067811865476
    tmp17 = tmp13 * tmp16
    tmp18 = tl.math.erf(tmp17)
    tmp19 = 1.0
    tmp20 = tmp18 + tmp19
    tmp21 = tmp15 * tmp20
    tmp22 = tmp20 * tmp14
    tmp23 = tmp13 * tmp13
    tmp24 = -0.5
    tmp25 = tmp23 * tmp24
    tmp26 = tl.exp(tmp25)
    tmp27 = 0.3989422804014327
    tmp28 = tmp26 * tmp27
    tmp29 = tmp13 * tmp28
    tmp30 = tmp22 + tmp29
    tl.store(out_ptr1 + (x2), tmp21, None)
    tl.store(out_ptr2 + (x2), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kp/ckpkcoivb5x724kuqigwijrrod5dklhjf6c76whqvpts5dl43qz7.py
# Source Nodes: [l__mod___patch_embed_proj_2_0], Original ATen: [aten.convolution]
# l__mod___patch_embed_proj_2_0 => convolution_1
triton_poi_fused_convolution_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (384*x2) + (1204224*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oh/cohiu4bydo3r4ehyvf6n2m4xloqemlqnp6xt3nsah6zfu6x4ijec.py
# Source Nodes: [l__mod___patch_embed_proj_2_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___patch_embed_proj_2_1 => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 75264
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zd/czdnnosnvksohupp3t4esld75lpylrenr2mdhs5kv3vrnw6afhov.py
# Source Nodes: [l__mod___patch_embed_proj_2_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___patch_embed_proj_2_1 => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (384*r2) + (37632*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (384*r2) + (37632*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (384*r2) + (37632*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (384*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (384*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (384*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7w/c7w524ewrzliutojgb5ms6mi2oozssh5merdszaewf7xihqnj6nf.py
# Source Nodes: [l__mod___patch_embed_proj_2_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___patch_embed_proj_2_1 => add_7, add_8, add_9, mul_11, mul_12, mul_13, mul_14, mul_15, rsqrt_1, squeeze_4, var_mean_1
triton_per_fused__native_batch_norm_legit_functional_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_12', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 25088.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000398612827361
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ei/ceiyk2d5alepkhx42jsi3tbqbylwqncbsgxanzl2dot474nw2idj.py
# Source Nodes: [l__mod___patch_embed_proj_2_1, l__mod___patch_embed_proj_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu, aten.gelu_backward]
# l__mod___patch_embed_proj_2_1 => add_10, add_7, mul_10, mul_16, rsqrt_1, sub_1, var_mean_1
# l__mod___patch_embed_proj_3 => add_11, erf_1, mul_17, mul_18, mul_19
triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9633792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.5
    tmp15 = tmp13 * tmp14
    tmp16 = 0.7071067811865476
    tmp17 = tmp13 * tmp16
    tmp18 = tl.math.erf(tmp17)
    tmp19 = 1.0
    tmp20 = tmp18 + tmp19
    tmp21 = tmp15 * tmp20
    tmp22 = tmp20 * tmp14
    tmp23 = tmp13 * tmp13
    tmp24 = -0.5
    tmp25 = tmp23 * tmp24
    tmp26 = tl.exp(tmp25)
    tmp27 = 0.3989422804014327
    tmp28 = tmp26 * tmp27
    tmp29 = tmp13 * tmp28
    tmp30 = tmp22 + tmp29
    tl.store(out_ptr1 + (x2), tmp21, None)
    tl.store(out_ptr2 + (x2), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dw/cdwhag2hzmtipu7mcad2mm7dk2v7rc6qbndrblbaf6rla2b7ctwd.py
# Source Nodes: [l__mod___patch_embed_proj_4_0], Original ATen: [aten.convolution]
# l__mod___patch_embed_proj_4_0 => convolution_2
triton_poi_fused_convolution_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (768*x2) + (602112*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vu/cvuckovyfo2tqmtm3ihw3ncg6fj4ypf4zec3nyfzcqyimaayxmpp.py
# Source Nodes: [x], Original ATen: [aten._native_batch_norm_legit_functional]
# x => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qc/cqcbal7e5eo6hujwaq2you6ankf3gbwyceyxvlrc63em3krigkmi.py
# Source Nodes: [x], Original ATen: [aten._native_batch_norm_legit_functional]
# x => add_13, add_14, add_15, mul_21, mul_22, mul_23, mul_24, mul_25, rsqrt_2, squeeze_7, var_mean_2
triton_per_fused__native_batch_norm_legit_functional_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_16', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 6272.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0001594642002871
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vn/cvncpra4cag2t2cjgnutaxrriapakxxe64oiz6dnca74qepx2rce.py
# Source Nodes: [stack_3], Original ATen: [aten.stack]
# stack_3 => cat
triton_poi_fused_stack_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x2 = (xindex // 32) % 28
    x1 = (xindex // 2) % 16
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8 + tmp7
    tmp10 = 28.000001907348633
    tmp11 = tmp9 / tmp10
    tmp12 = 6.283185307179586
    tmp13 = tmp11 * tmp12
    tmp14 = 2*x1
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 * tmp7
    tmp17 = 0.0
    tmp18 = tmp16 + tmp17
    tmp19 = 2.0
    tmp20 = tmp18 / tmp19
    tmp21 = tl.math.floor(tmp20)
    tmp22 = tmp21 * tmp19
    tmp23 = 32.0
    tmp24 = tmp22 / tmp23
    tmp25 = 10000.0
    tmp26 = tl.math.pow(tmp25, tmp24)
    tmp27 = tmp13 / tmp26
    tmp28 = tl.sin(tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 2, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = 1 + (2*x1)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp35 * tmp7
    tmp37 = tmp36 + tmp17
    tmp38 = tmp37 / tmp19
    tmp39 = tl.math.floor(tmp38)
    tmp40 = tmp39 * tmp19
    tmp41 = tmp40 / tmp23
    tmp42 = tl.math.pow(tmp25, tmp41)
    tmp43 = tmp13 / tmp42
    tmp44 = tl.cos(tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp31, tmp44, tmp45)
    tmp47 = tl.where(tmp4, tmp30, tmp46)
    tl.store(out_ptr0 + (x5), tmp47, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/op/copxhukx2yeq4b5pyjrovkhdhu7gpzv3mcsnunjzion3vpn3fqkr.py
# Source Nodes: [stack_2], Original ATen: [aten.stack]
# stack_2 => cat_1
triton_poi_fused_stack_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x3 = (xindex // 896)
    x1 = (xindex // 2) % 16
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8 + tmp7
    tmp10 = 28.000001907348633
    tmp11 = tmp9 / tmp10
    tmp12 = 6.283185307179586
    tmp13 = tmp11 * tmp12
    tmp14 = 2*x1
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 * tmp7
    tmp17 = 0.0
    tmp18 = tmp16 + tmp17
    tmp19 = 2.0
    tmp20 = tmp18 / tmp19
    tmp21 = tl.math.floor(tmp20)
    tmp22 = tmp21 * tmp19
    tmp23 = 32.0
    tmp24 = tmp22 / tmp23
    tmp25 = 10000.0
    tmp26 = tl.math.pow(tmp25, tmp24)
    tmp27 = tmp13 / tmp26
    tmp28 = tl.sin(tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 2, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = 1 + (2*x1)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp35 * tmp7
    tmp37 = tmp36 + tmp17
    tmp38 = tmp37 / tmp19
    tmp39 = tl.math.floor(tmp38)
    tmp40 = tmp39 * tmp19
    tmp41 = tmp40 / tmp23
    tmp42 = tl.math.pow(tmp25, tmp41)
    tmp43 = tmp13 / tmp42
    tmp44 = tl.cos(tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp31, tmp44, tmp45)
    tmp47 = tl.where(tmp4, tmp30, tmp46)
    tl.store(out_ptr0 + (x5), tmp47, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ja/cjaxqighanoe7ln764qbgwulziw2r5u7fv3asbqotiimpqni4tr5.py
# Source Nodes: [cat_11, pos], Original ATen: [aten.cat, aten.permute]
# cat_11 => cat_2
# pos => permute_1
triton_poi_fused_cat_permute_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_permute_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (32*x1)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 64, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-32) + x0 + (32*x1)), tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sb/csbhgp2e6dvtehqkadhw5wmyroklhz2ztycipi5qjzdn6uzhhjar.py
# Source Nodes: [l__mod___blocks_0_attn_qkv, l__mod___blocks_0_norm1, x_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_0_attn_qkv => view_4
# l__mod___blocks_0_norm1 => add_23, add_24, clone_1, mul_33, mul_34, rsqrt_3, sub_3, var_mean_3
# x_3 => add_22
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 784
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr5 + (x0 + (784*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 + tmp15
    tmp17 = tmp13 + tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tl.full([1], 768, tl.int32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 / tmp26
    tmp28 = tmp18 - tmp27
    tmp29 = tmp28 * tmp28
    tmp30 = tl.broadcast_to(tmp29, [RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp30, 0)
    tmp33 = triton_helpers.promote_to_tensor(tl.sum(tmp32, 0))
    tmp34 = tmp17 - tmp27
    tmp35 = 768.0
    tmp36 = tmp33 / tmp35
    tmp37 = 1e-06
    tmp38 = tmp36 + tmp37
    tmp39 = tl.math.rsqrt(tmp38)
    tmp40 = tmp34 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = tmp39 / tmp35
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp17, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp40, rmask & xmask)
    tl.store(out_ptr4 + (r2 + (768*x3)), tmp44, rmask & xmask)
    tl.store(out_ptr5 + (x3), tmp45, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ao/caow7aw24mv3pqtnvvdfwe6zofolrpacyvkkuyzhlyz55wxbn27r.py
# Source Nodes: [q_1], Original ATen: [aten.linalg_vector_norm]
# q_1 => pow_2, sum_1
triton_red_fused_linalg_vector_norm_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_linalg_vector_norm_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 43008
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2304*r2) + (258048*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(rmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hb/chbkohfcyj4zbdwrxtqdpr2ag6nzliv4feujn4zqh4hblx47r23i.py
# Source Nodes: [q_1], Original ATen: [aten.linalg_vector_norm]
# q_1 => pow_2, pow_3, sum_1
triton_per_fused_linalg_vector_norm_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_linalg_vector_norm_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 768
    x1 = (xindex // 768)
    x3 = xindex
    x4 = xindex % 48
    x5 = (xindex // 48) % 16
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (5376*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = tl.sqrt(tmp4)
    tl.store(out_ptr1 + (x5 + (16*x4) + (768*x1)), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7f/c7fkvijt4co3xxsbna4jw4tahtovn6bz3horoqbiqd3kzapnk72l.py
# Source Nodes: [k_1], Original ATen: [aten.linalg_vector_norm]
# k_1 => pow_4, sum_2
triton_red_fused_linalg_vector_norm_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_linalg_vector_norm_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 43008
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (768 + x0 + (2304*r2) + (258048*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(rmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wr/cwrmtrz7odmjguttedgvzdnvzeyqk6j4bp244kobp6pkarhgrzy3.py
# Source Nodes: [matmul, q_1], Original ATen: [aten.clone, aten.div]
# matmul => clone_2
# q_1 => div_6
triton_poi_fused_clone_div_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_div_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = (yindex // 768)
    y4 = yindex % 768
    y0 = yindex % 48
    y1 = (yindex // 48) % 16
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (y4 + (2304*x3) + (1806336*y2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1 + (16*y0) + (768*y2)), None, eviction_policy='evict_last')
    tmp2 = 1e-12
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tmp0 / tmp3
    tl.store(out_ptr0 + (x3 + (784*y5)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/js/cjs46knq24h34eqsl3m4fvrciuna6e3zvfr3thdv2zbes5ol5nep.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_3
triton_poi_fused_clone_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 48
    x1 = (xindex // 48) % 784
    x2 = (xindex // 37632) % 16
    x3 = (xindex // 602112)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x0 + (48*x2) + (2304*x1) + (1806336*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x2 + (16*x0) + (768*x3)), None, eviction_policy='evict_last')
    tmp2 = 1e-12
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tmp0 / tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7b/c7brahx6qy36v4vyyxll767wjiaoizbc5jb5tec65ujtx3byebpf.py
# Source Nodes: [attn, attn_1, attn_2], Original ATen: [aten._softmax, aten.clone, aten.mul]
# attn => mul_35
# attn_1 => amax, div_8, exp, sub_4, sum_3
# attn_2 => clone_4
triton_per_fused__softmax_clone_mul_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_mul_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 48
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x1 = (xindex // 48) % 16
    tmp0 = tl.load(in_ptr0 + (r3 + (48*x4)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr2 + (r3 + (48*x4)), tmp13, rmask)
    tl.store(out_ptr0 + (x4), tmp6, None)
    tl.store(out_ptr1 + (x4), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/77/c77mcpnovjr3npf7qkgqxohjxpzvylecsgxkj2aygneyw554oohj.py
# Source Nodes: [matmul_1], Original ATen: [aten.clone]
# matmul_1 => clone_5
triton_poi_fused_clone_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (1536 + y0 + (2304*x2) + (1806336*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lj/cljkeqnsok52a5fqds6axie5xsecjtye7s2ebvomvggq6rot7ami.py
# Source Nodes: [x_6], Original ATen: [aten._unsafe_view, aten.clone]
# x_6 => clone_6, view_14
triton_poi_fused__unsafe_view_clone_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((784*x1) + (602112*(y0 // 784)) + (y0 % 784)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (768*y0)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bw/cbwawr5j6bhwqz2gq5pycorpgba7xijqpo477ay3osjcmo7mk7bt.py
# Source Nodes: [l__mod___blocks_0_norm3, mul_4, x_6, x_8, x_9], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_0_norm3 => add_27, clone_8, mul_37, rsqrt_4, sub_5, var_mean_4
# mul_4 => mul_36
# x_6 => add_25
# x_8 => add_26
# x_9 => view_16
triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 768, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 768.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-06
    tmp27 = tmp25 + tmp26
    tmp28 = tl.math.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp28 / tmp24
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp33, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fa/cfa2q2g5vmgcrhz2bcivxizgg27qas6njgxvqsjesuvnibysfn7l.py
# Source Nodes: [x_10], Original ATen: [aten.convolution]
# x_10 => convolution_4
triton_poi_fused_convolution_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (768*x2) + (602112*y1)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yx/cyxnb333pku5qivlfrk2wbb6mx3yrtnybd2get25tnp7enh76m4z.py
# Source Nodes: [x_11, x_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
# x_11 => add_29, erf_2, mul_39, mul_40, mul_41
# x_12 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_gelu_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_gelu_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.5
        tmp2 = tmp0 * tmp1
        tmp3 = 0.7071067811865476
        tmp4 = tmp0 * tmp3
        tmp5 = tl.math.erf(tmp4)
        tmp6 = 1.0
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight,
        )
        tmp10_mean = tl.where(rmask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask & xmask, tmp10_weight_next, tmp10_weight)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp11, xmask)
    tl.store(out_ptr2 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7o/c7ovh3fxcb3ghdyanyhgibzpyfbrrkzhhxcivibfuq2mpgkekez4.py
# Source Nodes: [x_11, x_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
# x_11 => add_29, erf_2, mul_39, mul_40, mul_41
# x_12 => add_31, add_34, mul_42, mul_48, rsqrt_5, sub_6, var_mean_5
triton_poi_fused__native_batch_norm_legit_functional_gelu_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_gelu_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp9 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 6272.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp10 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tl.store(out_ptr0 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oa/coafzhc6pc4hf77owo4rviponcu2fqifte76uq63krp2bnqjn2kz.py
# Source Nodes: [l__mod___blocks_0_norm2, mul_4, mul_5, x_15, x_16, x_6, x_8], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_0_norm2 => add_36, add_37, clone_9, mul_50, mul_51, rsqrt_6, sub_7, var_mean_6
# mul_4 => mul_36
# mul_5 => mul_49
# x_15 => add_35
# x_16 => view_18
# x_6 => add_25
# x_8 => add_26
triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp34 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 768, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tmp10 - tmp20
    tmp28 = 768.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-06
    tmp31 = tmp29 + tmp30
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tmp32 / tmp28
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp10, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp33, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp37, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp38, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oq/coqgt73utyekaxlezexo32eyab4d2clp2l4bffr74ik6x2xxnhu3.py
# Source Nodes: [x_17, x_20], Original ATen: [aten.gelu, aten.view]
# x_17 => add_38, erf_3, mul_52, mul_53, mul_54
# x_20 => view_20
triton_poi_fused_gelu_view_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19267584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/o4/co4evngxqawybdtxzfxjfap5uui77e3oxkpcwnnjx3obalz726lt.py
# Source Nodes: [l__mod___blocks_1_attn_qkv, l__mod___blocks_1_norm1, mul_6, x_23], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_1_attn_qkv => view_22
# l__mod___blocks_1_norm1 => add_40, add_41, clone_12, mul_56, mul_57, rsqrt_7, sub_8, var_mean_7
# mul_6 => mul_55
# x_23 => add_39
triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5x/c5xetzqjq73mkpjhvydvhgxlta4ftxldcwtyzlqrmpif4jkxdr7y.py
# Source Nodes: [l__mod___blocks_1_norm3, mul_6, mul_8, x_23, x_25, x_27, x_28], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_1_norm3 => add_44, clone_19, mul_60, rsqrt_8, sub_10, var_mean_8
# mul_6 => mul_55
# mul_8 => mul_59
# x_23 => add_39
# x_25 => add_42
# x_27 => add_43
# x_28 => view_34
triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tmp4 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 768, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tmp10 - tmp20
    tmp28 = 768.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-06
    tmp31 = tmp29 + tmp30
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tmp32 / tmp28
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp10, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp33, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp37, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp38, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qo/cqodgzroikgzi337g57uaptpn4ztlqd35ow2outgllhoamjjjydh.py
# Source Nodes: [l__mod___blocks_2_attn_qkv, l__mod___blocks_2_norm1, mul_10, mul_9, x_34, x_42], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_2_attn_qkv => view_40
# l__mod___blocks_2_norm1 => add_57, add_58, clone_23, mul_79, mul_80, rsqrt_11, sub_13, var_mean_11
# mul_10 => mul_78
# mul_9 => mul_72
# x_34 => add_52
# x_42 => add_56
triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 768.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vz/cvznyx7xvfkhp5wgixuhatiojboon2ciigjzcptrenbxrr5lnbkg.py
# Source Nodes: [cat_10, x_norm1], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_10 => cat_3
# x_norm1 => add_431, add_432, mul_585, mul_586, rsqrt_99, sub_123, var_mean_99
triton_per_fused_cat_native_layer_norm_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_38', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 785
    r2 = rindex
    x1 = (xindex // 785)
    x3 = xindex
    tmp46 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 785, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-768) + r2 + (768*x0) + (602112*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr3 + (r2 + (768*(((-1) + x0) % 784)) + (602112*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = tl.load(in_ptr4 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr5 + ((-768) + r2 + (768*x0) + (602112*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp8, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = tl.full([1], 768, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp39 = 768.0
    tmp40 = tmp38 / tmp39
    tmp41 = 1e-06
    tmp42 = tmp40 + tmp41
    tmp43 = tl.math.rsqrt(tmp42)
    tmp44 = tmp22 - tmp32
    tmp45 = tmp44 * tmp43
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp22, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp43, xmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp49, rmask & xmask)
    tl.store(out_ptr1 + (x3), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ae/caeqsxzfx7y77csnjbk4zjpr4klpioao5ob74kn2mtx5yl4xvwll.py
# Source Nodes: [q_48, x_cls], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
# q_48 => permute_220
# x_cls => _scaled_dot_product_efficient_attention
triton_poi_fused__scaled_dot_product_efficient_attention_permute_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_permute_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 48
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
    tmp0 = tl.load(in_ptr0 + (x2 + (48*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (768*y1)), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (48*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2p/c2p5m2dqdyfs7jlcv5shjcmklfieare3uumg47jq67kxedlkra3g.py
# Source Nodes: [k_48, x_cls], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
# k_48 => permute_222
# x_cls => _scaled_dot_product_efficient_attention
triton_poi_fused__scaled_dot_product_efficient_attention_permute_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_permute_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100480
    xnumel = 48
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
    tmp0 = tl.load(in_ptr0 + (x2 + (48*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (768*y1)), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (48*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sy/csyvfv5q4bk3t6d65t2oclwinx5ki3njox6mjzg3c32einler2g5.py
# Source Nodes: [cat_9, mul_99, x_462, x_res], Original ATen: [aten.add, aten.cat, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# cat_9 => cat_4
# mul_99 => mul_587
# x_462 => add_433
# x_res => add_434, add_435, mul_588, mul_589, rsqrt_100, sub_124, var_mean_100
triton_per_fused_add_cat_mul_native_layer_norm_native_layer_norm_backward_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_mul_native_layer_norm_native_layer_norm_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 785
    r2 = rindex
    x1 = (xindex // 785)
    x3 = xindex
    tmp15 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (768*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 785, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + (r2 + (768*x3)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp17 = tmp16 * tmp14
    tmp18 = tmp15 + tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tl.full([1], 768, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = tmp18 - tmp28
    tmp36 = 768.0
    tmp37 = tmp34 / tmp36
    tmp38 = 1e-06
    tmp39 = tmp37 + tmp38
    tmp40 = tl.math.rsqrt(tmp39)
    tmp41 = tmp35 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = tmp40 / tmp36
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp14, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp41, rmask & xmask)
    tl.store(out_ptr4 + (r2 + (768*x3)), tmp45, rmask & xmask)
    tl.store(out_ptr5 + (x3), tmp46, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zf/czfe46inbcfhbkjl355zvyzurki52kq4e2zl4pukem3ruhvwynle.py
# Source Nodes: [x_464, x_465, x_468], Original ATen: [aten.add, aten.gelu, aten.view]
# x_464 => add_436
# x_465 => add_437, erf_50, mul_590, mul_591, mul_592
# x_468 => view_448
triton_poi_fused_add_gelu_view_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_view_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 3072
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vo/cvolghxd3tkcgyt4ccww5iakjyhanwhogozwwdi326xd72zbsvjl.py
# Source Nodes: [cat_8, x_472, x_norm1_1], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.native_layer_norm_backward]
# cat_8 => cat_5
# x_472 => add_438
# x_norm1_1 => add_439, add_440, mul_594, mul_595, rsqrt_101, sub_125, var_mean_101
triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 785
    x1 = (xindex // 785)
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp42 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = x0
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r2 + (768*x1)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = tmp1 >= tmp4
    tmp12 = tl.full([1], 785, tl.int64)
    tmp13 = tmp1 < tmp12
    tmp14 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & tmp11 & xmask, other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tl.where(tmp5, tmp10, tmp16)
    tmp18 = tmp0 + tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tl.full([1], 768, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = tmp18 - tmp28
    tmp36 = 768.0
    tmp37 = tmp34 / tmp36
    tmp38 = 1e-06
    tmp39 = tmp37 + tmp38
    tmp40 = tl.math.rsqrt(tmp39)
    tmp41 = tmp35 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = tmp40 / tmp36
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp41, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp45, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp46, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6g/c6g4zmtftjlwurhtezdhp6frgjopuh6imb7i4eylnuyvkvlvtod5.py
# Source Nodes: [cat_7, cat_8, mul_101, x_472, x_473, x_res_1], Original ATen: [aten.add, aten.cat, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# cat_7 => cat_6
# cat_8 => cat_5
# mul_101 => mul_596
# x_472 => add_438
# x_473 => add_441
# x_res_1 => add_442, add_443, mul_597, mul_598, rsqrt_102, sub_126, var_mean_102
triton_per_fused_add_cat_mul_native_layer_norm_native_layer_norm_backward_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_mul_native_layer_norm_native_layer_norm_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 785
    r2 = rindex
    x1 = (xindex // 785)
    x3 = xindex
    tmp15 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (768*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 785, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + (r2 + (768*x3)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr4 + (r2 + (768*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp4, tmp18, tmp19)
    tmp21 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask & tmp8 & xmask, other=0.0)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp8, tmp21, tmp22)
    tmp24 = tl.where(tmp4, tmp20, tmp23)
    tmp25 = tmp15 + tmp24
    tmp27 = tmp26 * tmp14
    tmp28 = tmp25 + tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = tl.broadcast_to(tmp29, [RBLOCK])
    tmp34 = tl.where(rmask & xmask, tmp32, 0)
    tmp35 = triton_helpers.promote_to_tensor(tl.sum(tmp34, 0))
    tmp36 = tl.full([1], 768, tl.int32)
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 / tmp37
    tmp39 = tmp29 - tmp38
    tmp40 = tmp39 * tmp39
    tmp41 = tl.broadcast_to(tmp40, [RBLOCK])
    tmp43 = tl.where(rmask & xmask, tmp41, 0)
    tmp44 = triton_helpers.promote_to_tensor(tl.sum(tmp43, 0))
    tmp45 = tmp28 - tmp38
    tmp46 = 768.0
    tmp47 = tmp44 / tmp46
    tmp48 = 1e-06
    tmp49 = tmp47 + tmp48
    tmp50 = tl.math.rsqrt(tmp49)
    tmp51 = tmp45 * tmp50
    tmp53 = tmp51 * tmp52
    tmp55 = tmp53 + tmp54
    tmp56 = tmp50 / tmp46
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp14, rmask & xmask)
    tl.store(out_ptr1 + (r2 + (768*x3)), tmp28, rmask & xmask)
    tl.store(out_ptr4 + (r2 + (768*x3)), tmp51, rmask & xmask)
    tl.store(out_ptr5 + (r2 + (768*x3)), tmp55, rmask & xmask)
    tl.store(out_ptr6 + (x3), tmp56, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a3/ca3jmykwtfyutvhl4pbsz7uobcthfevcloxqvudrapaak6cit5w3.py
# Source Nodes: [cat_6, x_483, x_485], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.native_layer_norm_backward]
# cat_6 => cat_7
# x_483 => add_446
# x_485 => add_447, mul_603, rsqrt_103, sub_127, var_mean_103
triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 785
    x1 = (xindex // 785)
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp1 = x0
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r2 + (768*x1)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = tmp1 >= tmp4
    tmp12 = tl.full([1], 785, tl.int64)
    tmp13 = tmp1 < tmp12
    tmp14 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & tmp11 & xmask, other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tl.where(tmp5, tmp10, tmp16)
    tmp18 = tmp0 + tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tl.full([1], 768, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = tmp18 - tmp28
    tmp36 = 768.0
    tmp37 = tmp34 / tmp36
    tmp38 = 1e-06
    tmp39 = tmp37 + tmp38
    tmp40 = tl.math.rsqrt(tmp39)
    tmp41 = tmp35 * tmp40
    tmp42 = tmp40 / tmp36
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp41, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp42, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7q/c7qaqbxgx4sczg7o6b76auam7qfr3mvdv6flp3ipm3npu2zjemmo.py
# Source Nodes: [x_487], Original ATen: [aten.clone]
# x_487 => clone_271
triton_poi_fused_clone_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (602880*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ef/cefs3mkv5mlsqgvkqv57bflgmlpbleo7efi27tbe7ughdlk6srbd.py
# Source Nodes: [], Original ATen: [aten.detach]

triton_poi_fused_detach_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_detach_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 48
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
    tmp0 = tl.load(in_ptr0 + (x2 + (48*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (768*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wp/cwp7dnltdwu2bvlebxj7a45uzw3t2xbdl7hop6bf7efjcrbg5e24.py
# Source Nodes: [attn_69, attn_70], Original ATen: [aten._softmax, aten.detach, aten.mul]
# attn_69 => mul_564
# attn_70 => div_77, exp_23, sub_119
triton_poi_fused__softmax_detach_mul_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_detach_mul_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 2304
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x5 = xindex
    y4 = yindex
    y0 = yindex % 16
    x3 = (xindex // 48)
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (x5 + (2304*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3 + (48*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3 + (48*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 - tmp3
    tmp5 = tl.exp(tmp4)
    tmp7 = tmp5 / tmp6
    tl.store(out_ptr0 + (y0 + (16*x5) + (36864*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cz/cczucca6ija2byy45bq3koq55y42bzyxqagmyk2fciddlq4tmpci.py
# Source Nodes: [l__mod___patch_embed_proj_0_1], Original ATen: [aten.add]
# l__mod___patch_embed_proj_0_1 => add
triton_poi_fused_add_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_49', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710 = args
    args.clear()
    assert_size_stride(primals_1, (768, ), (1, ))
    assert_size_stride(primals_2, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_22, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_26, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_34, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_38, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (768, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_46, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_50, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_58, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_66, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_70, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_74, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_78, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (768, ), (1, ))
    assert_size_stride(primals_81, (768, ), (1, ))
    assert_size_stride(primals_82, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (768, ), (1, ))
    assert_size_stride(primals_86, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_90, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_91, (768, ), (1, ))
    assert_size_stride(primals_92, (768, ), (1, ))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_94, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (768, ), (1, ))
    assert_size_stride(primals_97, (1, 1, 768), (768, 768, 1))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (768, ), (1, ))
    assert_size_stride(primals_102, (192, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_103, (192, ), (1, ))
    assert_size_stride(primals_104, (192, ), (1, ))
    assert_size_stride(primals_105, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_106, (384, ), (1, ))
    assert_size_stride(primals_107, (384, ), (1, ))
    assert_size_stride(primals_108, (768, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_109, (768, ), (1, ))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (768, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (768, ), (1, ))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (2304, 768), (768, 1))
    assert_size_stride(primals_116, (2304, ), (1, ))
    assert_size_stride(primals_117, (768, 768), (768, 1))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_122, (768, ), (1, ))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_124, (768, ), (1, ))
    assert_size_stride(primals_125, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_128, (768, ), (1, ))
    assert_size_stride(primals_129, (3072, 768), (768, 1))
    assert_size_stride(primals_130, (3072, ), (1, ))
    assert_size_stride(primals_131, (768, 3072), (3072, 1))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_133, (768, ), (1, ))
    assert_size_stride(primals_134, (768, ), (1, ))
    assert_size_stride(primals_135, (2304, 768), (768, 1))
    assert_size_stride(primals_136, (2304, ), (1, ))
    assert_size_stride(primals_137, (768, 768), (768, 1))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_140, (768, ), (1, ))
    assert_size_stride(primals_141, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_144, (768, ), (1, ))
    assert_size_stride(primals_145, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_146, (768, ), (1, ))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (3072, 768), (768, 1))
    assert_size_stride(primals_150, (3072, ), (1, ))
    assert_size_stride(primals_151, (768, 3072), (3072, 1))
    assert_size_stride(primals_152, (768, ), (1, ))
    assert_size_stride(primals_153, (768, ), (1, ))
    assert_size_stride(primals_154, (768, ), (1, ))
    assert_size_stride(primals_155, (2304, 768), (768, 1))
    assert_size_stride(primals_156, (2304, ), (1, ))
    assert_size_stride(primals_157, (768, 768), (768, 1))
    assert_size_stride(primals_158, (768, ), (1, ))
    assert_size_stride(primals_159, (768, ), (1, ))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_162, (768, ), (1, ))
    assert_size_stride(primals_163, (768, ), (1, ))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_165, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_166, (768, ), (1, ))
    assert_size_stride(primals_167, (768, ), (1, ))
    assert_size_stride(primals_168, (768, ), (1, ))
    assert_size_stride(primals_169, (3072, 768), (768, 1))
    assert_size_stride(primals_170, (3072, ), (1, ))
    assert_size_stride(primals_171, (768, 3072), (3072, 1))
    assert_size_stride(primals_172, (768, ), (1, ))
    assert_size_stride(primals_173, (768, ), (1, ))
    assert_size_stride(primals_174, (768, ), (1, ))
    assert_size_stride(primals_175, (2304, 768), (768, 1))
    assert_size_stride(primals_176, (2304, ), (1, ))
    assert_size_stride(primals_177, (768, 768), (768, 1))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_179, (768, ), (1, ))
    assert_size_stride(primals_180, (768, ), (1, ))
    assert_size_stride(primals_181, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_182, (768, ), (1, ))
    assert_size_stride(primals_183, (768, ), (1, ))
    assert_size_stride(primals_184, (768, ), (1, ))
    assert_size_stride(primals_185, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_186, (768, ), (1, ))
    assert_size_stride(primals_187, (768, ), (1, ))
    assert_size_stride(primals_188, (768, ), (1, ))
    assert_size_stride(primals_189, (3072, 768), (768, 1))
    assert_size_stride(primals_190, (3072, ), (1, ))
    assert_size_stride(primals_191, (768, 3072), (3072, 1))
    assert_size_stride(primals_192, (768, ), (1, ))
    assert_size_stride(primals_193, (768, ), (1, ))
    assert_size_stride(primals_194, (768, ), (1, ))
    assert_size_stride(primals_195, (2304, 768), (768, 1))
    assert_size_stride(primals_196, (2304, ), (1, ))
    assert_size_stride(primals_197, (768, 768), (768, 1))
    assert_size_stride(primals_198, (768, ), (1, ))
    assert_size_stride(primals_199, (768, ), (1, ))
    assert_size_stride(primals_200, (768, ), (1, ))
    assert_size_stride(primals_201, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_202, (768, ), (1, ))
    assert_size_stride(primals_203, (768, ), (1, ))
    assert_size_stride(primals_204, (768, ), (1, ))
    assert_size_stride(primals_205, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_206, (768, ), (1, ))
    assert_size_stride(primals_207, (768, ), (1, ))
    assert_size_stride(primals_208, (768, ), (1, ))
    assert_size_stride(primals_209, (3072, 768), (768, 1))
    assert_size_stride(primals_210, (3072, ), (1, ))
    assert_size_stride(primals_211, (768, 3072), (3072, 1))
    assert_size_stride(primals_212, (768, ), (1, ))
    assert_size_stride(primals_213, (768, ), (1, ))
    assert_size_stride(primals_214, (768, ), (1, ))
    assert_size_stride(primals_215, (2304, 768), (768, 1))
    assert_size_stride(primals_216, (2304, ), (1, ))
    assert_size_stride(primals_217, (768, 768), (768, 1))
    assert_size_stride(primals_218, (768, ), (1, ))
    assert_size_stride(primals_219, (768, ), (1, ))
    assert_size_stride(primals_220, (768, ), (1, ))
    assert_size_stride(primals_221, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_222, (768, ), (1, ))
    assert_size_stride(primals_223, (768, ), (1, ))
    assert_size_stride(primals_224, (768, ), (1, ))
    assert_size_stride(primals_225, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_226, (768, ), (1, ))
    assert_size_stride(primals_227, (768, ), (1, ))
    assert_size_stride(primals_228, (768, ), (1, ))
    assert_size_stride(primals_229, (3072, 768), (768, 1))
    assert_size_stride(primals_230, (3072, ), (1, ))
    assert_size_stride(primals_231, (768, 3072), (3072, 1))
    assert_size_stride(primals_232, (768, ), (1, ))
    assert_size_stride(primals_233, (768, ), (1, ))
    assert_size_stride(primals_234, (768, ), (1, ))
    assert_size_stride(primals_235, (2304, 768), (768, 1))
    assert_size_stride(primals_236, (2304, ), (1, ))
    assert_size_stride(primals_237, (768, 768), (768, 1))
    assert_size_stride(primals_238, (768, ), (1, ))
    assert_size_stride(primals_239, (768, ), (1, ))
    assert_size_stride(primals_240, (768, ), (1, ))
    assert_size_stride(primals_241, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_242, (768, ), (1, ))
    assert_size_stride(primals_243, (768, ), (1, ))
    assert_size_stride(primals_244, (768, ), (1, ))
    assert_size_stride(primals_245, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_246, (768, ), (1, ))
    assert_size_stride(primals_247, (768, ), (1, ))
    assert_size_stride(primals_248, (768, ), (1, ))
    assert_size_stride(primals_249, (3072, 768), (768, 1))
    assert_size_stride(primals_250, (3072, ), (1, ))
    assert_size_stride(primals_251, (768, 3072), (3072, 1))
    assert_size_stride(primals_252, (768, ), (1, ))
    assert_size_stride(primals_253, (768, ), (1, ))
    assert_size_stride(primals_254, (768, ), (1, ))
    assert_size_stride(primals_255, (2304, 768), (768, 1))
    assert_size_stride(primals_256, (2304, ), (1, ))
    assert_size_stride(primals_257, (768, 768), (768, 1))
    assert_size_stride(primals_258, (768, ), (1, ))
    assert_size_stride(primals_259, (768, ), (1, ))
    assert_size_stride(primals_260, (768, ), (1, ))
    assert_size_stride(primals_261, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_262, (768, ), (1, ))
    assert_size_stride(primals_263, (768, ), (1, ))
    assert_size_stride(primals_264, (768, ), (1, ))
    assert_size_stride(primals_265, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_266, (768, ), (1, ))
    assert_size_stride(primals_267, (768, ), (1, ))
    assert_size_stride(primals_268, (768, ), (1, ))
    assert_size_stride(primals_269, (3072, 768), (768, 1))
    assert_size_stride(primals_270, (3072, ), (1, ))
    assert_size_stride(primals_271, (768, 3072), (3072, 1))
    assert_size_stride(primals_272, (768, ), (1, ))
    assert_size_stride(primals_273, (768, ), (1, ))
    assert_size_stride(primals_274, (768, ), (1, ))
    assert_size_stride(primals_275, (2304, 768), (768, 1))
    assert_size_stride(primals_276, (2304, ), (1, ))
    assert_size_stride(primals_277, (768, 768), (768, 1))
    assert_size_stride(primals_278, (768, ), (1, ))
    assert_size_stride(primals_279, (768, ), (1, ))
    assert_size_stride(primals_280, (768, ), (1, ))
    assert_size_stride(primals_281, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_282, (768, ), (1, ))
    assert_size_stride(primals_283, (768, ), (1, ))
    assert_size_stride(primals_284, (768, ), (1, ))
    assert_size_stride(primals_285, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_286, (768, ), (1, ))
    assert_size_stride(primals_287, (768, ), (1, ))
    assert_size_stride(primals_288, (768, ), (1, ))
    assert_size_stride(primals_289, (3072, 768), (768, 1))
    assert_size_stride(primals_290, (3072, ), (1, ))
    assert_size_stride(primals_291, (768, 3072), (3072, 1))
    assert_size_stride(primals_292, (768, ), (1, ))
    assert_size_stride(primals_293, (768, ), (1, ))
    assert_size_stride(primals_294, (768, ), (1, ))
    assert_size_stride(primals_295, (2304, 768), (768, 1))
    assert_size_stride(primals_296, (2304, ), (1, ))
    assert_size_stride(primals_297, (768, 768), (768, 1))
    assert_size_stride(primals_298, (768, ), (1, ))
    assert_size_stride(primals_299, (768, ), (1, ))
    assert_size_stride(primals_300, (768, ), (1, ))
    assert_size_stride(primals_301, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_302, (768, ), (1, ))
    assert_size_stride(primals_303, (768, ), (1, ))
    assert_size_stride(primals_304, (768, ), (1, ))
    assert_size_stride(primals_305, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_306, (768, ), (1, ))
    assert_size_stride(primals_307, (768, ), (1, ))
    assert_size_stride(primals_308, (768, ), (1, ))
    assert_size_stride(primals_309, (3072, 768), (768, 1))
    assert_size_stride(primals_310, (3072, ), (1, ))
    assert_size_stride(primals_311, (768, 3072), (3072, 1))
    assert_size_stride(primals_312, (768, ), (1, ))
    assert_size_stride(primals_313, (768, ), (1, ))
    assert_size_stride(primals_314, (768, ), (1, ))
    assert_size_stride(primals_315, (2304, 768), (768, 1))
    assert_size_stride(primals_316, (2304, ), (1, ))
    assert_size_stride(primals_317, (768, 768), (768, 1))
    assert_size_stride(primals_318, (768, ), (1, ))
    assert_size_stride(primals_319, (768, ), (1, ))
    assert_size_stride(primals_320, (768, ), (1, ))
    assert_size_stride(primals_321, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_322, (768, ), (1, ))
    assert_size_stride(primals_323, (768, ), (1, ))
    assert_size_stride(primals_324, (768, ), (1, ))
    assert_size_stride(primals_325, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_326, (768, ), (1, ))
    assert_size_stride(primals_327, (768, ), (1, ))
    assert_size_stride(primals_328, (768, ), (1, ))
    assert_size_stride(primals_329, (3072, 768), (768, 1))
    assert_size_stride(primals_330, (3072, ), (1, ))
    assert_size_stride(primals_331, (768, 3072), (3072, 1))
    assert_size_stride(primals_332, (768, ), (1, ))
    assert_size_stride(primals_333, (768, ), (1, ))
    assert_size_stride(primals_334, (768, ), (1, ))
    assert_size_stride(primals_335, (2304, 768), (768, 1))
    assert_size_stride(primals_336, (2304, ), (1, ))
    assert_size_stride(primals_337, (768, 768), (768, 1))
    assert_size_stride(primals_338, (768, ), (1, ))
    assert_size_stride(primals_339, (768, ), (1, ))
    assert_size_stride(primals_340, (768, ), (1, ))
    assert_size_stride(primals_341, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_342, (768, ), (1, ))
    assert_size_stride(primals_343, (768, ), (1, ))
    assert_size_stride(primals_344, (768, ), (1, ))
    assert_size_stride(primals_345, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_346, (768, ), (1, ))
    assert_size_stride(primals_347, (768, ), (1, ))
    assert_size_stride(primals_348, (768, ), (1, ))
    assert_size_stride(primals_349, (3072, 768), (768, 1))
    assert_size_stride(primals_350, (3072, ), (1, ))
    assert_size_stride(primals_351, (768, 3072), (3072, 1))
    assert_size_stride(primals_352, (768, ), (1, ))
    assert_size_stride(primals_353, (768, ), (1, ))
    assert_size_stride(primals_354, (768, ), (1, ))
    assert_size_stride(primals_355, (2304, 768), (768, 1))
    assert_size_stride(primals_356, (2304, ), (1, ))
    assert_size_stride(primals_357, (768, 768), (768, 1))
    assert_size_stride(primals_358, (768, ), (1, ))
    assert_size_stride(primals_359, (768, ), (1, ))
    assert_size_stride(primals_360, (768, ), (1, ))
    assert_size_stride(primals_361, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_362, (768, ), (1, ))
    assert_size_stride(primals_363, (768, ), (1, ))
    assert_size_stride(primals_364, (768, ), (1, ))
    assert_size_stride(primals_365, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_366, (768, ), (1, ))
    assert_size_stride(primals_367, (768, ), (1, ))
    assert_size_stride(primals_368, (768, ), (1, ))
    assert_size_stride(primals_369, (3072, 768), (768, 1))
    assert_size_stride(primals_370, (3072, ), (1, ))
    assert_size_stride(primals_371, (768, 3072), (3072, 1))
    assert_size_stride(primals_372, (768, ), (1, ))
    assert_size_stride(primals_373, (768, ), (1, ))
    assert_size_stride(primals_374, (768, ), (1, ))
    assert_size_stride(primals_375, (2304, 768), (768, 1))
    assert_size_stride(primals_376, (2304, ), (1, ))
    assert_size_stride(primals_377, (768, 768), (768, 1))
    assert_size_stride(primals_378, (768, ), (1, ))
    assert_size_stride(primals_379, (768, ), (1, ))
    assert_size_stride(primals_380, (768, ), (1, ))
    assert_size_stride(primals_381, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_382, (768, ), (1, ))
    assert_size_stride(primals_383, (768, ), (1, ))
    assert_size_stride(primals_384, (768, ), (1, ))
    assert_size_stride(primals_385, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_386, (768, ), (1, ))
    assert_size_stride(primals_387, (768, ), (1, ))
    assert_size_stride(primals_388, (768, ), (1, ))
    assert_size_stride(primals_389, (3072, 768), (768, 1))
    assert_size_stride(primals_390, (3072, ), (1, ))
    assert_size_stride(primals_391, (768, 3072), (3072, 1))
    assert_size_stride(primals_392, (768, ), (1, ))
    assert_size_stride(primals_393, (768, ), (1, ))
    assert_size_stride(primals_394, (768, ), (1, ))
    assert_size_stride(primals_395, (2304, 768), (768, 1))
    assert_size_stride(primals_396, (2304, ), (1, ))
    assert_size_stride(primals_397, (768, 768), (768, 1))
    assert_size_stride(primals_398, (768, ), (1, ))
    assert_size_stride(primals_399, (768, ), (1, ))
    assert_size_stride(primals_400, (768, ), (1, ))
    assert_size_stride(primals_401, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_402, (768, ), (1, ))
    assert_size_stride(primals_403, (768, ), (1, ))
    assert_size_stride(primals_404, (768, ), (1, ))
    assert_size_stride(primals_405, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_406, (768, ), (1, ))
    assert_size_stride(primals_407, (768, ), (1, ))
    assert_size_stride(primals_408, (768, ), (1, ))
    assert_size_stride(primals_409, (3072, 768), (768, 1))
    assert_size_stride(primals_410, (3072, ), (1, ))
    assert_size_stride(primals_411, (768, 3072), (3072, 1))
    assert_size_stride(primals_412, (768, ), (1, ))
    assert_size_stride(primals_413, (768, ), (1, ))
    assert_size_stride(primals_414, (768, ), (1, ))
    assert_size_stride(primals_415, (2304, 768), (768, 1))
    assert_size_stride(primals_416, (2304, ), (1, ))
    assert_size_stride(primals_417, (768, 768), (768, 1))
    assert_size_stride(primals_418, (768, ), (1, ))
    assert_size_stride(primals_419, (768, ), (1, ))
    assert_size_stride(primals_420, (768, ), (1, ))
    assert_size_stride(primals_421, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_422, (768, ), (1, ))
    assert_size_stride(primals_423, (768, ), (1, ))
    assert_size_stride(primals_424, (768, ), (1, ))
    assert_size_stride(primals_425, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_426, (768, ), (1, ))
    assert_size_stride(primals_427, (768, ), (1, ))
    assert_size_stride(primals_428, (768, ), (1, ))
    assert_size_stride(primals_429, (3072, 768), (768, 1))
    assert_size_stride(primals_430, (3072, ), (1, ))
    assert_size_stride(primals_431, (768, 3072), (3072, 1))
    assert_size_stride(primals_432, (768, ), (1, ))
    assert_size_stride(primals_433, (768, ), (1, ))
    assert_size_stride(primals_434, (768, ), (1, ))
    assert_size_stride(primals_435, (2304, 768), (768, 1))
    assert_size_stride(primals_436, (2304, ), (1, ))
    assert_size_stride(primals_437, (768, 768), (768, 1))
    assert_size_stride(primals_438, (768, ), (1, ))
    assert_size_stride(primals_439, (768, ), (1, ))
    assert_size_stride(primals_440, (768, ), (1, ))
    assert_size_stride(primals_441, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_442, (768, ), (1, ))
    assert_size_stride(primals_443, (768, ), (1, ))
    assert_size_stride(primals_444, (768, ), (1, ))
    assert_size_stride(primals_445, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_446, (768, ), (1, ))
    assert_size_stride(primals_447, (768, ), (1, ))
    assert_size_stride(primals_448, (768, ), (1, ))
    assert_size_stride(primals_449, (3072, 768), (768, 1))
    assert_size_stride(primals_450, (3072, ), (1, ))
    assert_size_stride(primals_451, (768, 3072), (3072, 1))
    assert_size_stride(primals_452, (768, ), (1, ))
    assert_size_stride(primals_453, (768, ), (1, ))
    assert_size_stride(primals_454, (768, ), (1, ))
    assert_size_stride(primals_455, (2304, 768), (768, 1))
    assert_size_stride(primals_456, (2304, ), (1, ))
    assert_size_stride(primals_457, (768, 768), (768, 1))
    assert_size_stride(primals_458, (768, ), (1, ))
    assert_size_stride(primals_459, (768, ), (1, ))
    assert_size_stride(primals_460, (768, ), (1, ))
    assert_size_stride(primals_461, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_462, (768, ), (1, ))
    assert_size_stride(primals_463, (768, ), (1, ))
    assert_size_stride(primals_464, (768, ), (1, ))
    assert_size_stride(primals_465, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_466, (768, ), (1, ))
    assert_size_stride(primals_467, (768, ), (1, ))
    assert_size_stride(primals_468, (768, ), (1, ))
    assert_size_stride(primals_469, (3072, 768), (768, 1))
    assert_size_stride(primals_470, (3072, ), (1, ))
    assert_size_stride(primals_471, (768, 3072), (3072, 1))
    assert_size_stride(primals_472, (768, ), (1, ))
    assert_size_stride(primals_473, (768, ), (1, ))
    assert_size_stride(primals_474, (768, ), (1, ))
    assert_size_stride(primals_475, (2304, 768), (768, 1))
    assert_size_stride(primals_476, (2304, ), (1, ))
    assert_size_stride(primals_477, (768, 768), (768, 1))
    assert_size_stride(primals_478, (768, ), (1, ))
    assert_size_stride(primals_479, (768, ), (1, ))
    assert_size_stride(primals_480, (768, ), (1, ))
    assert_size_stride(primals_481, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_482, (768, ), (1, ))
    assert_size_stride(primals_483, (768, ), (1, ))
    assert_size_stride(primals_484, (768, ), (1, ))
    assert_size_stride(primals_485, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_486, (768, ), (1, ))
    assert_size_stride(primals_487, (768, ), (1, ))
    assert_size_stride(primals_488, (768, ), (1, ))
    assert_size_stride(primals_489, (3072, 768), (768, 1))
    assert_size_stride(primals_490, (3072, ), (1, ))
    assert_size_stride(primals_491, (768, 3072), (3072, 1))
    assert_size_stride(primals_492, (768, ), (1, ))
    assert_size_stride(primals_493, (768, ), (1, ))
    assert_size_stride(primals_494, (768, ), (1, ))
    assert_size_stride(primals_495, (2304, 768), (768, 1))
    assert_size_stride(primals_496, (2304, ), (1, ))
    assert_size_stride(primals_497, (768, 768), (768, 1))
    assert_size_stride(primals_498, (768, ), (1, ))
    assert_size_stride(primals_499, (768, ), (1, ))
    assert_size_stride(primals_500, (768, ), (1, ))
    assert_size_stride(primals_501, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_502, (768, ), (1, ))
    assert_size_stride(primals_503, (768, ), (1, ))
    assert_size_stride(primals_504, (768, ), (1, ))
    assert_size_stride(primals_505, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_506, (768, ), (1, ))
    assert_size_stride(primals_507, (768, ), (1, ))
    assert_size_stride(primals_508, (768, ), (1, ))
    assert_size_stride(primals_509, (3072, 768), (768, 1))
    assert_size_stride(primals_510, (3072, ), (1, ))
    assert_size_stride(primals_511, (768, 3072), (3072, 1))
    assert_size_stride(primals_512, (768, ), (1, ))
    assert_size_stride(primals_513, (768, ), (1, ))
    assert_size_stride(primals_514, (768, ), (1, ))
    assert_size_stride(primals_515, (2304, 768), (768, 1))
    assert_size_stride(primals_516, (2304, ), (1, ))
    assert_size_stride(primals_517, (768, 768), (768, 1))
    assert_size_stride(primals_518, (768, ), (1, ))
    assert_size_stride(primals_519, (768, ), (1, ))
    assert_size_stride(primals_520, (768, ), (1, ))
    assert_size_stride(primals_521, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_522, (768, ), (1, ))
    assert_size_stride(primals_523, (768, ), (1, ))
    assert_size_stride(primals_524, (768, ), (1, ))
    assert_size_stride(primals_525, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_526, (768, ), (1, ))
    assert_size_stride(primals_527, (768, ), (1, ))
    assert_size_stride(primals_528, (768, ), (1, ))
    assert_size_stride(primals_529, (3072, 768), (768, 1))
    assert_size_stride(primals_530, (3072, ), (1, ))
    assert_size_stride(primals_531, (768, 3072), (3072, 1))
    assert_size_stride(primals_532, (768, ), (1, ))
    assert_size_stride(primals_533, (768, ), (1, ))
    assert_size_stride(primals_534, (768, ), (1, ))
    assert_size_stride(primals_535, (2304, 768), (768, 1))
    assert_size_stride(primals_536, (2304, ), (1, ))
    assert_size_stride(primals_537, (768, 768), (768, 1))
    assert_size_stride(primals_538, (768, ), (1, ))
    assert_size_stride(primals_539, (768, ), (1, ))
    assert_size_stride(primals_540, (768, ), (1, ))
    assert_size_stride(primals_541, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_542, (768, ), (1, ))
    assert_size_stride(primals_543, (768, ), (1, ))
    assert_size_stride(primals_544, (768, ), (1, ))
    assert_size_stride(primals_545, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_546, (768, ), (1, ))
    assert_size_stride(primals_547, (768, ), (1, ))
    assert_size_stride(primals_548, (768, ), (1, ))
    assert_size_stride(primals_549, (3072, 768), (768, 1))
    assert_size_stride(primals_550, (3072, ), (1, ))
    assert_size_stride(primals_551, (768, 3072), (3072, 1))
    assert_size_stride(primals_552, (768, ), (1, ))
    assert_size_stride(primals_553, (768, ), (1, ))
    assert_size_stride(primals_554, (768, ), (1, ))
    assert_size_stride(primals_555, (2304, 768), (768, 1))
    assert_size_stride(primals_556, (2304, ), (1, ))
    assert_size_stride(primals_557, (768, 768), (768, 1))
    assert_size_stride(primals_558, (768, ), (1, ))
    assert_size_stride(primals_559, (768, ), (1, ))
    assert_size_stride(primals_560, (768, ), (1, ))
    assert_size_stride(primals_561, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_562, (768, ), (1, ))
    assert_size_stride(primals_563, (768, ), (1, ))
    assert_size_stride(primals_564, (768, ), (1, ))
    assert_size_stride(primals_565, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_566, (768, ), (1, ))
    assert_size_stride(primals_567, (768, ), (1, ))
    assert_size_stride(primals_568, (768, ), (1, ))
    assert_size_stride(primals_569, (3072, 768), (768, 1))
    assert_size_stride(primals_570, (3072, ), (1, ))
    assert_size_stride(primals_571, (768, 3072), (3072, 1))
    assert_size_stride(primals_572, (768, ), (1, ))
    assert_size_stride(primals_573, (768, ), (1, ))
    assert_size_stride(primals_574, (768, ), (1, ))
    assert_size_stride(primals_575, (2304, 768), (768, 1))
    assert_size_stride(primals_576, (2304, ), (1, ))
    assert_size_stride(primals_577, (768, 768), (768, 1))
    assert_size_stride(primals_578, (768, ), (1, ))
    assert_size_stride(primals_579, (768, ), (1, ))
    assert_size_stride(primals_580, (768, ), (1, ))
    assert_size_stride(primals_581, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_582, (768, ), (1, ))
    assert_size_stride(primals_583, (768, ), (1, ))
    assert_size_stride(primals_584, (768, ), (1, ))
    assert_size_stride(primals_585, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_586, (768, ), (1, ))
    assert_size_stride(primals_587, (768, ), (1, ))
    assert_size_stride(primals_588, (768, ), (1, ))
    assert_size_stride(primals_589, (3072, 768), (768, 1))
    assert_size_stride(primals_590, (3072, ), (1, ))
    assert_size_stride(primals_591, (768, 3072), (3072, 1))
    assert_size_stride(primals_592, (768, ), (1, ))
    assert_size_stride(primals_593, (768, ), (1, ))
    assert_size_stride(primals_594, (768, ), (1, ))
    assert_size_stride(primals_595, (768, 768), (768, 1))
    assert_size_stride(primals_596, (768, ), (1, ))
    assert_size_stride(primals_597, (768, 768), (768, 1))
    assert_size_stride(primals_598, (768, ), (1, ))
    assert_size_stride(primals_599, (768, 768), (768, 1))
    assert_size_stride(primals_600, (768, ), (1, ))
    assert_size_stride(primals_601, (768, 768), (768, 1))
    assert_size_stride(primals_602, (768, ), (1, ))
    assert_size_stride(primals_603, (768, ), (1, ))
    assert_size_stride(primals_604, (768, ), (1, ))
    assert_size_stride(primals_605, (3072, 768), (768, 1))
    assert_size_stride(primals_606, (3072, ), (1, ))
    assert_size_stride(primals_607, (768, 3072), (3072, 1))
    assert_size_stride(primals_608, (768, ), (1, ))
    assert_size_stride(primals_609, (768, ), (1, ))
    assert_size_stride(primals_610, (768, ), (1, ))
    assert_size_stride(primals_611, (768, 768), (768, 1))
    assert_size_stride(primals_612, (768, ), (1, ))
    assert_size_stride(primals_613, (768, 768), (768, 1))
    assert_size_stride(primals_614, (768, ), (1, ))
    assert_size_stride(primals_615, (768, 768), (768, 1))
    assert_size_stride(primals_616, (768, ), (1, ))
    assert_size_stride(primals_617, (768, 768), (768, 1))
    assert_size_stride(primals_618, (768, ), (1, ))
    assert_size_stride(primals_619, (768, ), (1, ))
    assert_size_stride(primals_620, (768, ), (1, ))
    assert_size_stride(primals_621, (3072, 768), (768, 1))
    assert_size_stride(primals_622, (3072, ), (1, ))
    assert_size_stride(primals_623, (768, 3072), (3072, 1))
    assert_size_stride(primals_624, (768, ), (1, ))
    assert_size_stride(primals_625, (768, ), (1, ))
    assert_size_stride(primals_626, (768, ), (1, ))
    assert_size_stride(primals_627, (1000, 768), (768, 1))
    assert_size_stride(primals_628, (1000, ), (1, ))
    assert_size_stride(primals_629, (192, ), (1, ))
    assert_size_stride(primals_630, (192, ), (1, ))
    assert_size_stride(primals_631, (), ())
    assert_size_stride(primals_632, (384, ), (1, ))
    assert_size_stride(primals_633, (384, ), (1, ))
    assert_size_stride(primals_634, (), ())
    assert_size_stride(primals_635, (768, ), (1, ))
    assert_size_stride(primals_636, (768, ), (1, ))
    assert_size_stride(primals_637, (), ())
    assert_size_stride(primals_638, (768, ), (1, ))
    assert_size_stride(primals_639, (768, ), (1, ))
    assert_size_stride(primals_640, (), ())
    assert_size_stride(primals_641, (768, ), (1, ))
    assert_size_stride(primals_642, (768, ), (1, ))
    assert_size_stride(primals_643, (), ())
    assert_size_stride(primals_644, (768, ), (1, ))
    assert_size_stride(primals_645, (768, ), (1, ))
    assert_size_stride(primals_646, (), ())
    assert_size_stride(primals_647, (768, ), (1, ))
    assert_size_stride(primals_648, (768, ), (1, ))
    assert_size_stride(primals_649, (), ())
    assert_size_stride(primals_650, (768, ), (1, ))
    assert_size_stride(primals_651, (768, ), (1, ))
    assert_size_stride(primals_652, (), ())
    assert_size_stride(primals_653, (768, ), (1, ))
    assert_size_stride(primals_654, (768, ), (1, ))
    assert_size_stride(primals_655, (), ())
    assert_size_stride(primals_656, (768, ), (1, ))
    assert_size_stride(primals_657, (768, ), (1, ))
    assert_size_stride(primals_658, (), ())
    assert_size_stride(primals_659, (768, ), (1, ))
    assert_size_stride(primals_660, (768, ), (1, ))
    assert_size_stride(primals_661, (), ())
    assert_size_stride(primals_662, (768, ), (1, ))
    assert_size_stride(primals_663, (768, ), (1, ))
    assert_size_stride(primals_664, (), ())
    assert_size_stride(primals_665, (768, ), (1, ))
    assert_size_stride(primals_666, (768, ), (1, ))
    assert_size_stride(primals_667, (), ())
    assert_size_stride(primals_668, (768, ), (1, ))
    assert_size_stride(primals_669, (768, ), (1, ))
    assert_size_stride(primals_670, (), ())
    assert_size_stride(primals_671, (768, ), (1, ))
    assert_size_stride(primals_672, (768, ), (1, ))
    assert_size_stride(primals_673, (), ())
    assert_size_stride(primals_674, (768, ), (1, ))
    assert_size_stride(primals_675, (768, ), (1, ))
    assert_size_stride(primals_676, (), ())
    assert_size_stride(primals_677, (768, ), (1, ))
    assert_size_stride(primals_678, (768, ), (1, ))
    assert_size_stride(primals_679, (), ())
    assert_size_stride(primals_680, (768, ), (1, ))
    assert_size_stride(primals_681, (768, ), (1, ))
    assert_size_stride(primals_682, (), ())
    assert_size_stride(primals_683, (768, ), (1, ))
    assert_size_stride(primals_684, (768, ), (1, ))
    assert_size_stride(primals_685, (), ())
    assert_size_stride(primals_686, (768, ), (1, ))
    assert_size_stride(primals_687, (768, ), (1, ))
    assert_size_stride(primals_688, (), ())
    assert_size_stride(primals_689, (768, ), (1, ))
    assert_size_stride(primals_690, (768, ), (1, ))
    assert_size_stride(primals_691, (), ())
    assert_size_stride(primals_692, (768, ), (1, ))
    assert_size_stride(primals_693, (768, ), (1, ))
    assert_size_stride(primals_694, (), ())
    assert_size_stride(primals_695, (768, ), (1, ))
    assert_size_stride(primals_696, (768, ), (1, ))
    assert_size_stride(primals_697, (), ())
    assert_size_stride(primals_698, (768, ), (1, ))
    assert_size_stride(primals_699, (768, ), (1, ))
    assert_size_stride(primals_700, (), ())
    assert_size_stride(primals_701, (768, ), (1, ))
    assert_size_stride(primals_702, (768, ), (1, ))
    assert_size_stride(primals_703, (), ())
    assert_size_stride(primals_704, (768, ), (1, ))
    assert_size_stride(primals_705, (768, ), (1, ))
    assert_size_stride(primals_706, (), ())
    assert_size_stride(primals_707, (768, ), (1, ))
    assert_size_stride(primals_708, (768, ), (1, ))
    assert_size_stride(primals_709, (), ())
    assert_size_stride(primals_710, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((192, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_102, buf0, 576, 9, grid=grid(576, 9), stream=stream0)
        del primals_102
        buf1 = empty_strided((384, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_105, buf1, 73728, 9, grid=grid(73728, 9), stream=stream0)
        del primals_105
        buf2 = empty_strided((768, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_108, buf2, 294912, 9, grid=grid(294912, 9), stream=stream0)
        del primals_108
        buf3 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_710, buf3, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del primals_710
        # Source Nodes: [l__mod___patch_embed_proj_0_0], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (8, 192, 112, 112), (2408448, 12544, 112, 1))
        buf5 = empty_strided((8, 192, 112, 112), (2408448, 1, 21504, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_proj_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_4.run(buf4, buf5, 1536, 12544, grid=grid(1536, 12544), stream=stream0)
        buf6 = empty_strided((1, 192, 1, 1, 512), (98304, 1, 98304, 98304, 192), device='cuda', dtype=torch.float32)
        buf7 = empty_strided((1, 192, 1, 1, 512), (98304, 1, 98304, 98304, 192), device='cuda', dtype=torch.float32)
        buf8 = empty_strided((1, 192, 1, 1, 512), (98304, 1, 98304, 98304, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_proj_0_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_5.run(buf5, buf6, buf7, buf8, 98304, 196, grid=grid(98304), stream=stream0)
        buf9 = empty_strided((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), device='cuda', dtype=torch.float32)
        buf10 = empty_strided((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), device='cuda', dtype=torch.float32)
        buf11 = empty_strided((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_proj_0_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_6.run(buf6, buf7, buf8, buf9, buf10, buf11, 768, 128, grid=grid(768), stream=stream0)
        del buf6
        del buf7
        del buf8
        buf12 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf13 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf15 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_proj_0_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_7.run(buf9, buf10, buf11, primals_629, primals_630, buf12, buf13, buf15, primals_629, primals_630, 192, 4, grid=grid(192), stream=stream0)
        del primals_629
        del primals_630
        buf17 = reinterpret_tensor(buf4, (8, 192, 112, 112), (2408448, 1, 21504, 192), 0); del buf4  # reuse
        buf1378 = empty_strided((8, 192, 112, 112), (2408448, 1, 21504, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_proj_0_1, l__mod___patch_embed_proj_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu, aten.gelu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_8.run(buf5, buf12, buf13, primals_103, primals_104, buf17, buf1378, 19267584, grid=grid(19267584), stream=stream0)
        del buf13
        del primals_104
        # Source Nodes: [l__mod___patch_embed_proj_2_0], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 384, 56, 56), (1204224, 3136, 56, 1))
        buf19 = empty_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_proj_2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(buf18, buf19, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        buf20 = empty_strided((1, 384, 1, 1, 196), (75264, 1, 75264, 75264, 384), device='cuda', dtype=torch.float32)
        buf21 = empty_strided((1, 384, 1, 1, 196), (75264, 1, 75264, 75264, 384), device='cuda', dtype=torch.float32)
        buf22 = empty_strided((1, 384, 1, 1, 196), (75264, 1, 75264, 75264, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_proj_2_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_10.run(buf19, buf20, buf21, buf22, 75264, 128, grid=grid(75264), stream=stream0)
        buf23 = reinterpret_tensor(buf9, (1, 384, 1, 1, 2), (768, 1, 768, 768, 384), 0); del buf9  # reuse
        buf24 = reinterpret_tensor(buf11, (1, 384, 1, 1, 2), (768, 1, 768, 768, 384), 0); del buf11  # reuse
        buf25 = reinterpret_tensor(buf10, (1, 384, 1, 1, 2), (768, 1, 768, 768, 384), 0); del buf10  # reuse
        # Source Nodes: [l__mod___patch_embed_proj_2_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf20, buf21, buf22, buf23, buf24, buf25, 768, 98, grid=grid(768), stream=stream0)
        del buf20
        del buf21
        del buf22
        buf26 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf27 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf29 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_proj_2_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_12.run(buf23, buf24, buf25, primals_632, primals_633, buf26, buf27, buf29, primals_632, primals_633, 384, 2, grid=grid(384), stream=stream0)
        del primals_632
        del primals_633
        buf31 = reinterpret_tensor(buf18, (8, 384, 56, 56), (1204224, 1, 21504, 384), 0); del buf18  # reuse
        buf1377 = empty_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_proj_2_1, l__mod___patch_embed_proj_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu, aten.gelu_backward]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_13.run(buf19, buf26, buf27, primals_106, primals_107, buf31, buf1377, 9633792, grid=grid(9633792), stream=stream0)
        del buf27
        del primals_107
        # Source Nodes: [l__mod___patch_embed_proj_4_0], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, buf2, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf33 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_proj_4_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf32, buf33, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf34 = empty_strided((1, 768, 1, 1, 49), (37632, 1, 37632, 37632, 768), device='cuda', dtype=torch.float32)
        buf35 = empty_strided((1, 768, 1, 1, 49), (37632, 1, 37632, 37632, 768), device='cuda', dtype=torch.float32)
        buf36 = empty_strided((1, 768, 1, 1, 49), (37632, 1, 37632, 37632, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf33, buf34, buf35, buf36, 37632, 128, grid=grid(37632), stream=stream0)
        buf37 = reinterpret_tensor(buf25, (1, 768, 1, 1), (768, 1, 768, 768), 0); del buf25  # reuse
        buf38 = reinterpret_tensor(buf24, (1, 768, 1, 1), (768, 1, 768, 768), 0); del buf24  # reuse
        buf40 = reinterpret_tensor(buf23, (768, ), (1, ), 0); del buf23  # reuse
        # Source Nodes: [x], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf34, buf35, buf36, primals_635, primals_636, buf37, buf38, buf40, primals_635, primals_636, 768, 49, grid=grid(768), stream=stream0)
        del primals_635
        del primals_636
        buf41 = empty((1, 28, 28, 16, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [stack_3], Original ATen: [aten.stack]
        triton_poi_fused_stack_17.run(buf41, 25088, grid=grid(25088), stream=stream0)
        buf42 = empty((1, 28, 28, 16, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [stack_2], Original ATen: [aten.stack]
        triton_poi_fused_stack_18.run(buf42, 25088, grid=grid(25088), stream=stream0)
        buf43 = empty_strided((1, 64, 28, 28), (50176, 1, 1792, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_11, pos], Original ATen: [aten.cat, aten.permute]
        triton_poi_fused_cat_permute_19.run(buf42, buf41, buf43, 50176, grid=grid(50176), stream=stream0)
        del buf41
        del buf42
        # Source Nodes: [pos_1], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (1, 768, 28, 28), (602112, 784, 28, 1))
        buf45 = reinterpret_tensor(buf32, (8, 784, 768), (602112, 768, 1), 0); del buf32  # reuse
        buf49 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf50 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1376 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_attn_qkv, l__mod___blocks_0_norm1, x_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20.run(buf33, buf37, buf38, primals_109, primals_110, buf44, primals_112, primals_113, primals_114, buf45, buf49, buf50, buf1376, 6272, 768, grid=grid(6272), stream=stream0)
        del buf44
        del primals_110
        del primals_112
        del primals_114
        buf51 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_116, buf50, reinterpret_tensor(primals_115, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf51)
        del primals_116
        buf52 = empty_strided((8, 16, 48, 1, 7), (5376, 48, 1, 43008, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_1], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf51, buf52, 43008, 112, grid=grid(43008), stream=stream0)
        buf54 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_1], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf52, buf54, 6144, 7, grid=grid(6144), stream=stream0)
        buf55 = buf52; del buf52  # reuse
        # Source Nodes: [k_1], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf51, buf55, 43008, 112, grid=grid(43008), stream=stream0)
        buf57 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_1], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf55, buf57, 6144, 7, grid=grid(6144), stream=stream0)
        buf58 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul, q_1], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf51, buf54, buf58, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf59 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf51, buf57, buf59, 4816896, grid=grid(4816896), stream=stream0)
        buf60 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf58, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf59, (128, 784, 48), (37632, 48, 1), 0), out=buf60)
        buf61 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf62 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf63 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn, attn_1, attn_2], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf60, primals_2, buf61, buf62, buf63, 6144, 48, grid=grid(6144), stream=stream0)
        buf64 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf51, buf64, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf65 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf63, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf64, (128, 48, 784), (37632, 784, 1), 0), out=buf65)
        buf66 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf65, buf66, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf67 = reinterpret_tensor(buf65, (6272, 768), (768, 1), 0); del buf65  # reuse
        # Source Nodes: [x_6], Original ATen: [aten.mm]
        extern_kernels.mm(buf66, reinterpret_tensor(primals_117, (768, 768), (1, 768), 0), out=buf67)
        buf71 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf72 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1374 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_norm3, mul_4, x_6, x_8, x_9], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_29.run(buf45, primals_1, buf67, primals_118, primals_119, primals_120, buf71, buf72, buf1374, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_120
        # Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, primals_121, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf73, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf74 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf73, primals_122, buf74, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_122
        buf75 = buf36; del buf36  # reuse
        buf76 = buf35; del buf35  # reuse
        buf77 = buf34; del buf34  # reuse
        # Source Nodes: [x_11, x_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf74, buf75, buf76, buf77, 37632, 128, grid=grid(37632), stream=stream0)
        buf78 = buf38; del buf38  # reuse
        buf79 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf81 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11, x_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf75, buf76, buf77, primals_638, primals_639, buf78, buf79, buf81, primals_638, primals_639, 768, 49, grid=grid(768), stream=stream0)
        del primals_638
        del primals_639
        buf82 = reinterpret_tensor(buf73, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf73  # reuse
        # Source Nodes: [x_11, x_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf74, buf78, buf79, primals_123, primals_124, buf82, 4816896, grid=grid(4816896), stream=stream0)
        del primals_124
        # Source Nodes: [x_13], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_125, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf83, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf84 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf83, primals_126, buf84, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_126
        buf85 = reinterpret_tensor(buf83, (8, 784, 768), (602112, 768, 1), 0); del buf83  # reuse
        buf89 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf90 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1373 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_norm2, mul_4, mul_5, x_15, x_16, x_6, x_8], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_33.run(buf45, primals_1, buf67, primals_118, primals_3, buf84, primals_127, primals_128, buf85, buf89, buf90, buf1373, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_128
        buf91 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_130, buf90, reinterpret_tensor(primals_129, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf91)
        del primals_130
        buf92 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17, x_20], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf91, buf92, 19267584, grid=grid(19267584), stream=stream0)
        buf93 = reinterpret_tensor(buf45, (6272, 768), (768, 1), 0); del buf45  # reuse
        # Source Nodes: [x_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_132, buf92, reinterpret_tensor(primals_131, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf93)
        del primals_132
        buf97 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf98 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1372 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_attn_qkv, l__mod___blocks_1_norm1, mul_6, x_23], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf85, primals_4, buf93, primals_133, primals_134, buf97, buf98, buf1372, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_134
        buf99 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_136, buf98, reinterpret_tensor(primals_135, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf99)
        del primals_136
        buf100 = buf55; del buf55  # reuse
        # Source Nodes: [q_3], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf99, buf100, 43008, 112, grid=grid(43008), stream=stream0)
        buf102 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_3], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf100, buf102, 6144, 7, grid=grid(6144), stream=stream0)
        buf103 = buf100; del buf100  # reuse
        # Source Nodes: [k_3], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf99, buf103, 43008, 112, grid=grid(43008), stream=stream0)
        buf105 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_3], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf103, buf105, 6144, 7, grid=grid(6144), stream=stream0)
        buf106 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2, q_3], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf99, buf102, buf106, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf107 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf99, buf105, buf107, 4816896, grid=grid(4816896), stream=stream0)
        buf108 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf106, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf107, (128, 784, 48), (37632, 48, 1), 0), out=buf108)
        buf109 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf110 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf111 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_3, attn_4, attn_5], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf108, primals_6, buf109, buf110, buf111, 6144, 48, grid=grid(6144), stream=stream0)
        buf112 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf99, buf112, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf113 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf111, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf112, (128, 48, 784), (37632, 784, 1), 0), out=buf113)
        buf114 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf113, buf114, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf115 = reinterpret_tensor(buf113, (6272, 768), (768, 1), 0); del buf113  # reuse
        # Source Nodes: [x_25], Original ATen: [aten.mm]
        extern_kernels.mm(buf114, reinterpret_tensor(primals_137, (768, 768), (1, 768), 0), out=buf115)
        buf116 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf120 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf121 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1370 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_norm3, mul_6, mul_8, x_23, x_25, x_27, x_28], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_36.run(buf85, primals_4, buf93, primals_5, buf115, primals_138, primals_139, primals_140, buf116, buf120, buf121, buf1370, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_140
        # Source Nodes: [x_29], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, primals_141, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf122, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf123 = reinterpret_tensor(buf85, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf85  # reuse
        # Source Nodes: [x_29], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf122, primals_142, buf123, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_142
        buf124 = buf77; del buf77  # reuse
        buf125 = buf76; del buf76  # reuse
        buf126 = buf75; del buf75  # reuse
        # Source Nodes: [x_30, x_31], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf123, buf124, buf125, buf126, 37632, 128, grid=grid(37632), stream=stream0)
        buf127 = buf79; del buf79  # reuse
        buf128 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf130 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_30, x_31], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf124, buf125, buf126, primals_641, primals_642, buf127, buf128, buf130, primals_641, primals_642, 768, 49, grid=grid(768), stream=stream0)
        del primals_641
        del primals_642
        buf131 = reinterpret_tensor(buf122, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf122  # reuse
        # Source Nodes: [x_30, x_31], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf123, buf127, buf128, primals_143, primals_144, buf131, 4816896, grid=grid(4816896), stream=stream0)
        del primals_144
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, primals_145, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf132, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf133 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf132, primals_146, buf133, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_146
        buf137 = reinterpret_tensor(buf132, (8, 784, 768), (602112, 768, 1), 0); del buf132  # reuse
        buf138 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1369 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_norm2, mul_9, x_34, x_35], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf116, primals_7, buf133, primals_147, primals_148, buf137, buf138, buf1369, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_148
        buf139 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_35], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_150, buf138, reinterpret_tensor(primals_149, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf139)
        del primals_150
        buf140 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_36, x_39], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf139, buf140, 19267584, grid=grid(19267584), stream=stream0)
        buf141 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_152, buf140, reinterpret_tensor(primals_151, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf141)
        del primals_152
        buf142 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf146 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf147 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1368 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_attn_qkv, l__mod___blocks_2_norm1, mul_10, mul_9, x_34, x_42], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_37.run(buf116, primals_7, buf133, primals_8, buf141, primals_153, primals_154, buf142, buf146, buf147, buf1368, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_154
        buf148 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_156, buf147, reinterpret_tensor(primals_155, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf148)
        del primals_156
        buf149 = buf103; del buf103  # reuse
        # Source Nodes: [q_5], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf148, buf149, 43008, 112, grid=grid(43008), stream=stream0)
        buf151 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_5], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf149, buf151, 6144, 7, grid=grid(6144), stream=stream0)
        buf152 = buf149; del buf149  # reuse
        # Source Nodes: [k_5], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf148, buf152, 43008, 112, grid=grid(43008), stream=stream0)
        buf154 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_5], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf152, buf154, 6144, 7, grid=grid(6144), stream=stream0)
        buf155 = reinterpret_tensor(buf116, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf116  # reuse
        # Source Nodes: [matmul_4, q_5], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf148, buf151, buf155, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf156 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf148, buf154, buf156, 4816896, grid=grid(4816896), stream=stream0)
        buf157 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf155, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf156, (128, 784, 48), (37632, 48, 1), 0), out=buf157)
        buf158 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf159 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf160 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_6, attn_7, attn_8], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf157, primals_10, buf158, buf159, buf160, 6144, 48, grid=grid(6144), stream=stream0)
        buf161 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf148, buf161, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf162 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf160, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf161, (128, 48, 784), (37632, 784, 1), 0), out=buf162)
        buf163 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf162, buf163, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf164 = reinterpret_tensor(buf162, (6272, 768), (768, 1), 0); del buf162  # reuse
        # Source Nodes: [x_44], Original ATen: [aten.mm]
        extern_kernels.mm(buf163, reinterpret_tensor(primals_157, (768, 768), (1, 768), 0), out=buf164)
        buf168 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf169 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1366 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_norm3, mul_12, x_44, x_46, x_47], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_29.run(buf142, primals_9, buf164, primals_158, primals_159, primals_160, buf168, buf169, buf1366, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_160
        # Source Nodes: [x_48], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, primals_161, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf170, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf171 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_48], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf170, primals_162, buf171, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_162
        buf172 = buf126; del buf126  # reuse
        buf173 = buf125; del buf125  # reuse
        buf174 = buf124; del buf124  # reuse
        # Source Nodes: [x_49, x_50], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf171, buf172, buf173, buf174, 37632, 128, grid=grid(37632), stream=stream0)
        buf175 = buf128; del buf128  # reuse
        buf176 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf178 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_49, x_50], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf172, buf173, buf174, primals_644, primals_645, buf175, buf176, buf178, primals_644, primals_645, 768, 49, grid=grid(768), stream=stream0)
        del primals_644
        del primals_645
        buf179 = reinterpret_tensor(buf170, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf170  # reuse
        # Source Nodes: [x_49, x_50], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf171, buf175, buf176, primals_163, primals_164, buf179, 4816896, grid=grid(4816896), stream=stream0)
        del primals_164
        # Source Nodes: [x_51], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf179, primals_165, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf180, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf181 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_51], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf180, primals_166, buf181, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_166
        buf182 = reinterpret_tensor(buf180, (8, 784, 768), (602112, 768, 1), 0); del buf180  # reuse
        buf186 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf187 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1365 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_norm2, mul_12, mul_13, x_44, x_46, x_53, x_54], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_33.run(buf142, primals_9, buf164, primals_158, primals_11, buf181, primals_167, primals_168, buf182, buf186, buf187, buf1365, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_168
        buf188 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_170, buf187, reinterpret_tensor(primals_169, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf188)
        del primals_170
        buf189 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_55, x_58], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf188, buf189, 19267584, grid=grid(19267584), stream=stream0)
        buf190 = reinterpret_tensor(buf142, (6272, 768), (768, 1), 0); del buf142  # reuse
        # Source Nodes: [x_58], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_172, buf189, reinterpret_tensor(primals_171, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf190)
        del primals_172
        buf194 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf195 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1364 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_attn_qkv, l__mod___blocks_3_norm1, mul_14, x_61], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf182, primals_12, buf190, primals_173, primals_174, buf194, buf195, buf1364, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_174
        buf196 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_176, buf195, reinterpret_tensor(primals_175, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf196)
        del primals_176
        buf197 = buf152; del buf152  # reuse
        # Source Nodes: [q_7], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf196, buf197, 43008, 112, grid=grid(43008), stream=stream0)
        buf199 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_7], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf197, buf199, 6144, 7, grid=grid(6144), stream=stream0)
        buf200 = buf197; del buf197  # reuse
        # Source Nodes: [k_7], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf196, buf200, 43008, 112, grid=grid(43008), stream=stream0)
        buf202 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_7], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf200, buf202, 6144, 7, grid=grid(6144), stream=stream0)
        buf203 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_6, q_7], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf196, buf199, buf203, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf204 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf196, buf202, buf204, 4816896, grid=grid(4816896), stream=stream0)
        buf205 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf203, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf204, (128, 784, 48), (37632, 48, 1), 0), out=buf205)
        buf206 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf207 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf208 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_10, attn_11, attn_9], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf205, primals_14, buf206, buf207, buf208, 6144, 48, grid=grid(6144), stream=stream0)
        buf209 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf196, buf209, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf210 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf208, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf209, (128, 48, 784), (37632, 784, 1), 0), out=buf210)
        buf211 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf210, buf211, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf212 = reinterpret_tensor(buf210, (6272, 768), (768, 1), 0); del buf210  # reuse
        # Source Nodes: [x_63], Original ATen: [aten.mm]
        extern_kernels.mm(buf211, reinterpret_tensor(primals_177, (768, 768), (1, 768), 0), out=buf212)
        buf213 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf217 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf218 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1362 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_norm3, mul_14, mul_16, x_61, x_63, x_65, x_66], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_36.run(buf182, primals_12, buf190, primals_13, buf212, primals_178, primals_179, primals_180, buf213, buf217, buf218, buf1362, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_180
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, primals_181, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf219, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf220 = reinterpret_tensor(buf182, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf182  # reuse
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf219, primals_182, buf220, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_182
        buf221 = buf174; del buf174  # reuse
        buf222 = buf173; del buf173  # reuse
        buf223 = buf172; del buf172  # reuse
        # Source Nodes: [x_68, x_69], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf220, buf221, buf222, buf223, 37632, 128, grid=grid(37632), stream=stream0)
        buf224 = buf176; del buf176  # reuse
        buf225 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf227 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68, x_69], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf221, buf222, buf223, primals_647, primals_648, buf224, buf225, buf227, primals_647, primals_648, 768, 49, grid=grid(768), stream=stream0)
        del primals_647
        del primals_648
        buf228 = reinterpret_tensor(buf219, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf219  # reuse
        # Source Nodes: [x_68, x_69], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf220, buf224, buf225, primals_183, primals_184, buf228, 4816896, grid=grid(4816896), stream=stream0)
        del primals_184
        # Source Nodes: [x_70], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, primals_185, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf229, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf230 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_70], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf229, primals_186, buf230, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_186
        buf234 = reinterpret_tensor(buf229, (8, 784, 768), (602112, 768, 1), 0); del buf229  # reuse
        buf235 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1361 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_norm2, mul_17, x_72, x_73], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf213, primals_15, buf230, primals_187, primals_188, buf234, buf235, buf1361, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_188
        buf236 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_73], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_190, buf235, reinterpret_tensor(primals_189, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf236)
        del primals_190
        buf237 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74, x_77], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf236, buf237, 19267584, grid=grid(19267584), stream=stream0)
        buf238 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_77], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_192, buf237, reinterpret_tensor(primals_191, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf238)
        del primals_192
        buf239 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf243 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf244 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1360 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_4_attn_qkv, l__mod___blocks_4_norm1, mul_17, mul_18, x_72, x_80], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_37.run(buf213, primals_15, buf230, primals_16, buf238, primals_193, primals_194, buf239, buf243, buf244, buf1360, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_194
        buf245 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_4_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_196, buf244, reinterpret_tensor(primals_195, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf245)
        del primals_196
        buf246 = buf200; del buf200  # reuse
        # Source Nodes: [q_9], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf245, buf246, 43008, 112, grid=grid(43008), stream=stream0)
        buf248 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_9], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf246, buf248, 6144, 7, grid=grid(6144), stream=stream0)
        buf249 = buf246; del buf246  # reuse
        # Source Nodes: [k_9], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf245, buf249, 43008, 112, grid=grid(43008), stream=stream0)
        buf251 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_9], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf249, buf251, 6144, 7, grid=grid(6144), stream=stream0)
        buf252 = reinterpret_tensor(buf213, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf213  # reuse
        # Source Nodes: [matmul_8, q_9], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf245, buf248, buf252, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf253 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf245, buf251, buf253, 4816896, grid=grid(4816896), stream=stream0)
        buf254 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf252, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf253, (128, 784, 48), (37632, 48, 1), 0), out=buf254)
        buf255 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf256 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf257 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_12, attn_13, attn_14], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf254, primals_18, buf255, buf256, buf257, 6144, 48, grid=grid(6144), stream=stream0)
        buf258 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf245, buf258, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf259 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf257, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf258, (128, 48, 784), (37632, 784, 1), 0), out=buf259)
        buf260 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_82], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf259, buf260, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf261 = reinterpret_tensor(buf259, (6272, 768), (768, 1), 0); del buf259  # reuse
        # Source Nodes: [x_82], Original ATen: [aten.mm]
        extern_kernels.mm(buf260, reinterpret_tensor(primals_197, (768, 768), (1, 768), 0), out=buf261)
        buf265 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf266 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1358 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_4_norm3, mul_20, x_82, x_84, x_85], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_29.run(buf239, primals_17, buf261, primals_198, primals_199, primals_200, buf265, buf266, buf1358, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_200
        # Source Nodes: [x_86], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf266, primals_201, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf267, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf268 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_86], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf267, primals_202, buf268, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_202
        buf269 = buf223; del buf223  # reuse
        buf270 = buf222; del buf222  # reuse
        buf271 = buf221; del buf221  # reuse
        # Source Nodes: [x_87, x_88], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf268, buf269, buf270, buf271, 37632, 128, grid=grid(37632), stream=stream0)
        buf272 = buf225; del buf225  # reuse
        buf273 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf275 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_87, x_88], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf269, buf270, buf271, primals_650, primals_651, buf272, buf273, buf275, primals_650, primals_651, 768, 49, grid=grid(768), stream=stream0)
        del primals_650
        del primals_651
        buf276 = reinterpret_tensor(buf267, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf267  # reuse
        # Source Nodes: [x_87, x_88], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf268, buf272, buf273, primals_203, primals_204, buf276, 4816896, grid=grid(4816896), stream=stream0)
        del primals_204
        # Source Nodes: [x_89], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, primals_205, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf277, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf278 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf277, primals_206, buf278, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_206
        buf279 = reinterpret_tensor(buf277, (8, 784, 768), (602112, 768, 1), 0); del buf277  # reuse
        buf283 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf284 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1357 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_4_norm2, mul_20, mul_21, x_82, x_84, x_91, x_92], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_33.run(buf239, primals_17, buf261, primals_198, primals_19, buf278, primals_207, primals_208, buf279, buf283, buf284, buf1357, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_208
        buf285 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_92], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_210, buf284, reinterpret_tensor(primals_209, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf285)
        del primals_210
        buf286 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_93, x_96], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf285, buf286, 19267584, grid=grid(19267584), stream=stream0)
        buf287 = reinterpret_tensor(buf239, (6272, 768), (768, 1), 0); del buf239  # reuse
        # Source Nodes: [x_96], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_212, buf286, reinterpret_tensor(primals_211, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf287)
        del primals_212
        buf291 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf292 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1356 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_5_attn_qkv, l__mod___blocks_5_norm1, mul_22, x_99], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf279, primals_20, buf287, primals_213, primals_214, buf291, buf292, buf1356, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_214
        buf293 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_5_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_216, buf292, reinterpret_tensor(primals_215, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf293)
        del primals_216
        buf294 = buf249; del buf249  # reuse
        # Source Nodes: [q_11], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf293, buf294, 43008, 112, grid=grid(43008), stream=stream0)
        buf296 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_11], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf294, buf296, 6144, 7, grid=grid(6144), stream=stream0)
        buf297 = buf294; del buf294  # reuse
        # Source Nodes: [k_11], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf293, buf297, 43008, 112, grid=grid(43008), stream=stream0)
        buf299 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_11], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf297, buf299, 6144, 7, grid=grid(6144), stream=stream0)
        buf300 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_10, q_11], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf293, buf296, buf300, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf301 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf293, buf299, buf301, 4816896, grid=grid(4816896), stream=stream0)
        buf302 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf300, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf301, (128, 784, 48), (37632, 48, 1), 0), out=buf302)
        buf303 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf304 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf305 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_15, attn_16, attn_17], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf302, primals_22, buf303, buf304, buf305, 6144, 48, grid=grid(6144), stream=stream0)
        buf306 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf293, buf306, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf307 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf305, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf306, (128, 48, 784), (37632, 784, 1), 0), out=buf307)
        buf308 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf307, buf308, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf309 = reinterpret_tensor(buf307, (6272, 768), (768, 1), 0); del buf307  # reuse
        # Source Nodes: [x_101], Original ATen: [aten.mm]
        extern_kernels.mm(buf308, reinterpret_tensor(primals_217, (768, 768), (1, 768), 0), out=buf309)
        buf310 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf314 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf315 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1354 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_5_norm3, mul_22, mul_24, x_101, x_103, x_104, x_99], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_36.run(buf279, primals_20, buf287, primals_21, buf309, primals_218, primals_219, primals_220, buf310, buf314, buf315, buf1354, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_220
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf315, primals_221, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf316, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf317 = reinterpret_tensor(buf279, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf279  # reuse
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf316, primals_222, buf317, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_222
        buf318 = buf271; del buf271  # reuse
        buf319 = buf270; del buf270  # reuse
        buf320 = buf269; del buf269  # reuse
        # Source Nodes: [x_106, x_107], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf317, buf318, buf319, buf320, 37632, 128, grid=grid(37632), stream=stream0)
        buf321 = buf273; del buf273  # reuse
        buf322 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf324 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106, x_107], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf318, buf319, buf320, primals_653, primals_654, buf321, buf322, buf324, primals_653, primals_654, 768, 49, grid=grid(768), stream=stream0)
        del primals_653
        del primals_654
        buf325 = reinterpret_tensor(buf316, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf316  # reuse
        # Source Nodes: [x_106, x_107], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf317, buf321, buf322, primals_223, primals_224, buf325, 4816896, grid=grid(4816896), stream=stream0)
        del primals_224
        # Source Nodes: [x_108], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf325, primals_225, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf326, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf327 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_108], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf326, primals_226, buf327, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_226
        buf331 = reinterpret_tensor(buf326, (8, 784, 768), (602112, 768, 1), 0); del buf326  # reuse
        buf332 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1353 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_5_norm2, mul_25, x_110, x_111], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf310, primals_23, buf327, primals_227, primals_228, buf331, buf332, buf1353, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_228
        buf333 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_111], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_230, buf332, reinterpret_tensor(primals_229, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf333)
        del primals_230
        buf334 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_112, x_115], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf333, buf334, 19267584, grid=grid(19267584), stream=stream0)
        buf335 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_115], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_232, buf334, reinterpret_tensor(primals_231, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf335)
        del primals_232
        buf336 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf340 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf341 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1352 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_6_attn_qkv, l__mod___blocks_6_norm1, mul_25, mul_26, x_110, x_118], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_37.run(buf310, primals_23, buf327, primals_24, buf335, primals_233, primals_234, buf336, buf340, buf341, buf1352, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_234
        buf342 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_6_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_236, buf341, reinterpret_tensor(primals_235, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf342)
        del primals_236
        buf343 = buf297; del buf297  # reuse
        # Source Nodes: [q_13], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf342, buf343, 43008, 112, grid=grid(43008), stream=stream0)
        buf345 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_13], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf343, buf345, 6144, 7, grid=grid(6144), stream=stream0)
        buf346 = buf343; del buf343  # reuse
        # Source Nodes: [k_13], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf342, buf346, 43008, 112, grid=grid(43008), stream=stream0)
        buf348 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_13], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf346, buf348, 6144, 7, grid=grid(6144), stream=stream0)
        buf349 = reinterpret_tensor(buf310, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf310  # reuse
        # Source Nodes: [matmul_12, q_13], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf342, buf345, buf349, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf350 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf342, buf348, buf350, 4816896, grid=grid(4816896), stream=stream0)
        buf351 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf349, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf350, (128, 784, 48), (37632, 48, 1), 0), out=buf351)
        buf352 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf353 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf354 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_18, attn_19, attn_20], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf351, primals_26, buf352, buf353, buf354, 6144, 48, grid=grid(6144), stream=stream0)
        buf355 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf342, buf355, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf356 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf354, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf355, (128, 48, 784), (37632, 784, 1), 0), out=buf356)
        buf357 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_120], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf356, buf357, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf358 = reinterpret_tensor(buf356, (6272, 768), (768, 1), 0); del buf356  # reuse
        # Source Nodes: [x_120], Original ATen: [aten.mm]
        extern_kernels.mm(buf357, reinterpret_tensor(primals_237, (768, 768), (1, 768), 0), out=buf358)
        buf362 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf363 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1350 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_6_norm3, mul_28, x_120, x_122, x_123], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_29.run(buf336, primals_25, buf358, primals_238, primals_239, primals_240, buf362, buf363, buf1350, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_240
        # Source Nodes: [x_124], Original ATen: [aten.convolution]
        buf364 = extern_kernels.convolution(buf363, primals_241, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf364, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf365 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_124], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf364, primals_242, buf365, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_242
        buf366 = buf320; del buf320  # reuse
        buf367 = buf319; del buf319  # reuse
        buf368 = buf318; del buf318  # reuse
        # Source Nodes: [x_125, x_126], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf365, buf366, buf367, buf368, 37632, 128, grid=grid(37632), stream=stream0)
        buf369 = buf322; del buf322  # reuse
        buf370 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf372 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_125, x_126], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf366, buf367, buf368, primals_656, primals_657, buf369, buf370, buf372, primals_656, primals_657, 768, 49, grid=grid(768), stream=stream0)
        del primals_656
        del primals_657
        buf373 = reinterpret_tensor(buf364, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf364  # reuse
        # Source Nodes: [x_125, x_126], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf365, buf369, buf370, primals_243, primals_244, buf373, 4816896, grid=grid(4816896), stream=stream0)
        del primals_244
        # Source Nodes: [x_127], Original ATen: [aten.convolution]
        buf374 = extern_kernels.convolution(buf373, primals_245, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf374, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf375 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_127], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf374, primals_246, buf375, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_246
        buf376 = reinterpret_tensor(buf374, (8, 784, 768), (602112, 768, 1), 0); del buf374  # reuse
        buf380 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf381 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1349 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_6_norm2, mul_28, mul_29, x_120, x_122, x_129, x_130], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_33.run(buf336, primals_25, buf358, primals_238, primals_27, buf375, primals_247, primals_248, buf376, buf380, buf381, buf1349, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_248
        buf382 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_130], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_250, buf381, reinterpret_tensor(primals_249, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf382)
        del primals_250
        buf383 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_131, x_134], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf382, buf383, 19267584, grid=grid(19267584), stream=stream0)
        buf384 = reinterpret_tensor(buf336, (6272, 768), (768, 1), 0); del buf336  # reuse
        # Source Nodes: [x_134], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_252, buf383, reinterpret_tensor(primals_251, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf384)
        del primals_252
        buf388 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf389 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1348 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_7_attn_qkv, l__mod___blocks_7_norm1, mul_30, x_137], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf376, primals_28, buf384, primals_253, primals_254, buf388, buf389, buf1348, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_254
        buf390 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_7_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_256, buf389, reinterpret_tensor(primals_255, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf390)
        del primals_256
        buf391 = buf346; del buf346  # reuse
        # Source Nodes: [q_15], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf390, buf391, 43008, 112, grid=grid(43008), stream=stream0)
        buf393 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_15], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf391, buf393, 6144, 7, grid=grid(6144), stream=stream0)
        buf394 = buf391; del buf391  # reuse
        # Source Nodes: [k_15], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf390, buf394, 43008, 112, grid=grid(43008), stream=stream0)
        buf396 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_15], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf394, buf396, 6144, 7, grid=grid(6144), stream=stream0)
        buf397 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_14, q_15], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf390, buf393, buf397, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf398 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf390, buf396, buf398, 4816896, grid=grid(4816896), stream=stream0)
        buf399 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf397, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf398, (128, 784, 48), (37632, 48, 1), 0), out=buf399)
        buf400 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf401 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf402 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_21, attn_22, attn_23], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf399, primals_30, buf400, buf401, buf402, 6144, 48, grid=grid(6144), stream=stream0)
        buf403 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf390, buf403, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf404 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf402, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf403, (128, 48, 784), (37632, 784, 1), 0), out=buf404)
        buf405 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_139], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf404, buf405, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf406 = reinterpret_tensor(buf404, (6272, 768), (768, 1), 0); del buf404  # reuse
        # Source Nodes: [x_139], Original ATen: [aten.mm]
        extern_kernels.mm(buf405, reinterpret_tensor(primals_257, (768, 768), (1, 768), 0), out=buf406)
        buf407 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf411 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf412 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1346 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_7_norm3, mul_30, mul_32, x_137, x_139, x_141, x_142], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_36.run(buf376, primals_28, buf384, primals_29, buf406, primals_258, primals_259, primals_260, buf407, buf411, buf412, buf1346, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_260
        # Source Nodes: [x_143], Original ATen: [aten.convolution]
        buf413 = extern_kernels.convolution(buf412, primals_261, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf413, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf414 = reinterpret_tensor(buf376, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf376  # reuse
        # Source Nodes: [x_143], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf413, primals_262, buf414, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_262
        buf415 = buf368; del buf368  # reuse
        buf416 = buf367; del buf367  # reuse
        buf417 = buf366; del buf366  # reuse
        # Source Nodes: [x_144, x_145], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf414, buf415, buf416, buf417, 37632, 128, grid=grid(37632), stream=stream0)
        buf418 = buf370; del buf370  # reuse
        buf419 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf421 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144, x_145], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf415, buf416, buf417, primals_659, primals_660, buf418, buf419, buf421, primals_659, primals_660, 768, 49, grid=grid(768), stream=stream0)
        del primals_659
        del primals_660
        buf422 = reinterpret_tensor(buf413, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf413  # reuse
        # Source Nodes: [x_144, x_145], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf414, buf418, buf419, primals_263, primals_264, buf422, 4816896, grid=grid(4816896), stream=stream0)
        del primals_264
        # Source Nodes: [x_146], Original ATen: [aten.convolution]
        buf423 = extern_kernels.convolution(buf422, primals_265, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf423, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf424 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_146], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf423, primals_266, buf424, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_266
        buf428 = reinterpret_tensor(buf423, (8, 784, 768), (602112, 768, 1), 0); del buf423  # reuse
        buf429 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1345 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_7_norm2, mul_33, x_148, x_149], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf407, primals_31, buf424, primals_267, primals_268, buf428, buf429, buf1345, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_268
        buf430 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_149], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_270, buf429, reinterpret_tensor(primals_269, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf430)
        del primals_270
        buf431 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_150, x_153], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf430, buf431, 19267584, grid=grid(19267584), stream=stream0)
        buf432 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_153], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_272, buf431, reinterpret_tensor(primals_271, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf432)
        del primals_272
        buf433 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf437 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf438 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1344 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_8_attn_qkv, l__mod___blocks_8_norm1, mul_33, mul_34, x_148, x_156], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_37.run(buf407, primals_31, buf424, primals_32, buf432, primals_273, primals_274, buf433, buf437, buf438, buf1344, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_274
        buf439 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_8_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_276, buf438, reinterpret_tensor(primals_275, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf439)
        del primals_276
        buf440 = buf394; del buf394  # reuse
        # Source Nodes: [q_17], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf439, buf440, 43008, 112, grid=grid(43008), stream=stream0)
        buf442 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_17], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf440, buf442, 6144, 7, grid=grid(6144), stream=stream0)
        buf443 = buf440; del buf440  # reuse
        # Source Nodes: [k_17], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf439, buf443, 43008, 112, grid=grid(43008), stream=stream0)
        buf445 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_17], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf443, buf445, 6144, 7, grid=grid(6144), stream=stream0)
        buf446 = reinterpret_tensor(buf407, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf407  # reuse
        # Source Nodes: [matmul_16, q_17], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf439, buf442, buf446, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf447 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf439, buf445, buf447, 4816896, grid=grid(4816896), stream=stream0)
        buf448 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf446, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf447, (128, 784, 48), (37632, 48, 1), 0), out=buf448)
        buf449 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf450 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf451 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_24, attn_25, attn_26], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf448, primals_34, buf449, buf450, buf451, 6144, 48, grid=grid(6144), stream=stream0)
        buf452 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf439, buf452, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf453 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf451, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf452, (128, 48, 784), (37632, 784, 1), 0), out=buf453)
        buf454 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf453, buf454, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf455 = reinterpret_tensor(buf453, (6272, 768), (768, 1), 0); del buf453  # reuse
        # Source Nodes: [x_158], Original ATen: [aten.mm]
        extern_kernels.mm(buf454, reinterpret_tensor(primals_277, (768, 768), (1, 768), 0), out=buf455)
        buf459 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf460 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1342 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_8_norm3, mul_36, x_158, x_160, x_161], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_29.run(buf433, primals_33, buf455, primals_278, primals_279, primals_280, buf459, buf460, buf1342, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_280
        # Source Nodes: [x_162], Original ATen: [aten.convolution]
        buf461 = extern_kernels.convolution(buf460, primals_281, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf461, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf462 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_162], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf461, primals_282, buf462, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_282
        buf463 = buf417; del buf417  # reuse
        buf464 = buf416; del buf416  # reuse
        buf465 = buf415; del buf415  # reuse
        # Source Nodes: [x_163, x_164], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf462, buf463, buf464, buf465, 37632, 128, grid=grid(37632), stream=stream0)
        buf466 = buf419; del buf419  # reuse
        buf467 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf469 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_163, x_164], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf463, buf464, buf465, primals_662, primals_663, buf466, buf467, buf469, primals_662, primals_663, 768, 49, grid=grid(768), stream=stream0)
        del primals_662
        del primals_663
        buf470 = reinterpret_tensor(buf461, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf461  # reuse
        # Source Nodes: [x_163, x_164], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf462, buf466, buf467, primals_283, primals_284, buf470, 4816896, grid=grid(4816896), stream=stream0)
        del primals_284
        # Source Nodes: [x_165], Original ATen: [aten.convolution]
        buf471 = extern_kernels.convolution(buf470, primals_285, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf471, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf472 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_165], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf471, primals_286, buf472, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_286
        buf473 = reinterpret_tensor(buf471, (8, 784, 768), (602112, 768, 1), 0); del buf471  # reuse
        buf477 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf478 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1341 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_8_norm2, mul_36, mul_37, x_158, x_160, x_167, x_168], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_33.run(buf433, primals_33, buf455, primals_278, primals_35, buf472, primals_287, primals_288, buf473, buf477, buf478, buf1341, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_288
        buf479 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_168], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_290, buf478, reinterpret_tensor(primals_289, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf479)
        del primals_290
        buf480 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_169, x_172], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf479, buf480, 19267584, grid=grid(19267584), stream=stream0)
        buf481 = reinterpret_tensor(buf433, (6272, 768), (768, 1), 0); del buf433  # reuse
        # Source Nodes: [x_172], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_292, buf480, reinterpret_tensor(primals_291, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf481)
        del primals_292
        buf485 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf486 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1340 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_9_attn_qkv, l__mod___blocks_9_norm1, mul_38, x_175], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf473, primals_36, buf481, primals_293, primals_294, buf485, buf486, buf1340, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_294
        buf487 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_9_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_296, buf486, reinterpret_tensor(primals_295, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf487)
        del primals_296
        buf488 = buf443; del buf443  # reuse
        # Source Nodes: [q_19], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf487, buf488, 43008, 112, grid=grid(43008), stream=stream0)
        buf490 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_19], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf488, buf490, 6144, 7, grid=grid(6144), stream=stream0)
        buf491 = buf488; del buf488  # reuse
        # Source Nodes: [k_19], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf487, buf491, 43008, 112, grid=grid(43008), stream=stream0)
        buf493 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_19], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf491, buf493, 6144, 7, grid=grid(6144), stream=stream0)
        buf494 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_18, q_19], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf487, buf490, buf494, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf495 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf487, buf493, buf495, 4816896, grid=grid(4816896), stream=stream0)
        buf496 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf494, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf495, (128, 784, 48), (37632, 48, 1), 0), out=buf496)
        buf497 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf498 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf499 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_27, attn_28, attn_29], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf496, primals_38, buf497, buf498, buf499, 6144, 48, grid=grid(6144), stream=stream0)
        buf500 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf487, buf500, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf501 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf499, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf500, (128, 48, 784), (37632, 784, 1), 0), out=buf501)
        buf502 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_177], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf501, buf502, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf503 = reinterpret_tensor(buf501, (6272, 768), (768, 1), 0); del buf501  # reuse
        # Source Nodes: [x_177], Original ATen: [aten.mm]
        extern_kernels.mm(buf502, reinterpret_tensor(primals_297, (768, 768), (1, 768), 0), out=buf503)
        buf504 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf508 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf509 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1338 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_9_norm3, mul_38, mul_40, x_175, x_177, x_179, x_180], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_36.run(buf473, primals_36, buf481, primals_37, buf503, primals_298, primals_299, primals_300, buf504, buf508, buf509, buf1338, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_300
        # Source Nodes: [x_181], Original ATen: [aten.convolution]
        buf510 = extern_kernels.convolution(buf509, primals_301, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf510, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf511 = reinterpret_tensor(buf473, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf473  # reuse
        # Source Nodes: [x_181], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf510, primals_302, buf511, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_302
        buf512 = buf465; del buf465  # reuse
        buf513 = buf464; del buf464  # reuse
        buf514 = buf463; del buf463  # reuse
        # Source Nodes: [x_182, x_183], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf511, buf512, buf513, buf514, 37632, 128, grid=grid(37632), stream=stream0)
        buf515 = buf467; del buf467  # reuse
        buf516 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf518 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_182, x_183], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf512, buf513, buf514, primals_665, primals_666, buf515, buf516, buf518, primals_665, primals_666, 768, 49, grid=grid(768), stream=stream0)
        del primals_665
        del primals_666
        buf519 = reinterpret_tensor(buf510, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf510  # reuse
        # Source Nodes: [x_182, x_183], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf511, buf515, buf516, primals_303, primals_304, buf519, 4816896, grid=grid(4816896), stream=stream0)
        del primals_304
        # Source Nodes: [x_184], Original ATen: [aten.convolution]
        buf520 = extern_kernels.convolution(buf519, primals_305, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf520, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf521 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_184], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf520, primals_306, buf521, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_306
        buf525 = reinterpret_tensor(buf520, (8, 784, 768), (602112, 768, 1), 0); del buf520  # reuse
        buf526 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1337 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_9_norm2, mul_41, x_186, x_187], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf504, primals_39, buf521, primals_307, primals_308, buf525, buf526, buf1337, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_308
        buf527 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_187], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_310, buf526, reinterpret_tensor(primals_309, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf527)
        del primals_310
        buf528 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188, x_191], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf527, buf528, 19267584, grid=grid(19267584), stream=stream0)
        buf529 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_191], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_312, buf528, reinterpret_tensor(primals_311, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf529)
        del primals_312
        buf530 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf534 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf535 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1336 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_10_attn_qkv, l__mod___blocks_10_norm1, mul_41, mul_42, x_186, x_194], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_37.run(buf504, primals_39, buf521, primals_40, buf529, primals_313, primals_314, buf530, buf534, buf535, buf1336, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_314
        buf536 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_10_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_316, buf535, reinterpret_tensor(primals_315, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf536)
        del primals_316
        buf537 = buf491; del buf491  # reuse
        # Source Nodes: [q_21], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf536, buf537, 43008, 112, grid=grid(43008), stream=stream0)
        buf539 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_21], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf537, buf539, 6144, 7, grid=grid(6144), stream=stream0)
        buf540 = buf537; del buf537  # reuse
        # Source Nodes: [k_21], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf536, buf540, 43008, 112, grid=grid(43008), stream=stream0)
        buf542 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_21], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf540, buf542, 6144, 7, grid=grid(6144), stream=stream0)
        buf543 = reinterpret_tensor(buf504, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf504  # reuse
        # Source Nodes: [matmul_20, q_21], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf536, buf539, buf543, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf544 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf536, buf542, buf544, 4816896, grid=grid(4816896), stream=stream0)
        buf545 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf543, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf544, (128, 784, 48), (37632, 48, 1), 0), out=buf545)
        buf546 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf547 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf548 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_30, attn_31, attn_32], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf545, primals_42, buf546, buf547, buf548, 6144, 48, grid=grid(6144), stream=stream0)
        buf549 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf536, buf549, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf550 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf548, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf549, (128, 48, 784), (37632, 784, 1), 0), out=buf550)
        buf551 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_196], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf550, buf551, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf552 = reinterpret_tensor(buf550, (6272, 768), (768, 1), 0); del buf550  # reuse
        # Source Nodes: [x_196], Original ATen: [aten.mm]
        extern_kernels.mm(buf551, reinterpret_tensor(primals_317, (768, 768), (1, 768), 0), out=buf552)
        buf556 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf557 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1334 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_10_norm3, mul_44, x_196, x_198, x_199], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_29.run(buf530, primals_41, buf552, primals_318, primals_319, primals_320, buf556, buf557, buf1334, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_320
        # Source Nodes: [x_200], Original ATen: [aten.convolution]
        buf558 = extern_kernels.convolution(buf557, primals_321, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf558, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf559 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_200], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf558, primals_322, buf559, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_322
        buf560 = buf514; del buf514  # reuse
        buf561 = buf513; del buf513  # reuse
        buf562 = buf512; del buf512  # reuse
        # Source Nodes: [x_201, x_202], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf559, buf560, buf561, buf562, 37632, 128, grid=grid(37632), stream=stream0)
        buf563 = buf516; del buf516  # reuse
        buf564 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf566 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_201, x_202], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf560, buf561, buf562, primals_668, primals_669, buf563, buf564, buf566, primals_668, primals_669, 768, 49, grid=grid(768), stream=stream0)
        del primals_668
        del primals_669
        buf567 = reinterpret_tensor(buf558, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf558  # reuse
        # Source Nodes: [x_201, x_202], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf559, buf563, buf564, primals_323, primals_324, buf567, 4816896, grid=grid(4816896), stream=stream0)
        del primals_324
        # Source Nodes: [x_203], Original ATen: [aten.convolution]
        buf568 = extern_kernels.convolution(buf567, primals_325, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf568, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf569 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_203], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf568, primals_326, buf569, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_326
        buf570 = reinterpret_tensor(buf568, (8, 784, 768), (602112, 768, 1), 0); del buf568  # reuse
        buf574 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf575 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1333 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_10_norm2, mul_44, mul_45, x_196, x_198, x_205, x_206], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_33.run(buf530, primals_41, buf552, primals_318, primals_43, buf569, primals_327, primals_328, buf570, buf574, buf575, buf1333, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_328
        buf576 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_206], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_330, buf575, reinterpret_tensor(primals_329, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf576)
        del primals_330
        buf577 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_207, x_210], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf576, buf577, 19267584, grid=grid(19267584), stream=stream0)
        buf578 = reinterpret_tensor(buf530, (6272, 768), (768, 1), 0); del buf530  # reuse
        # Source Nodes: [x_210], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_332, buf577, reinterpret_tensor(primals_331, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf578)
        del primals_332
        buf582 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf583 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1332 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_11_attn_qkv, l__mod___blocks_11_norm1, mul_46, x_213], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf570, primals_44, buf578, primals_333, primals_334, buf582, buf583, buf1332, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_334
        buf584 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_11_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_336, buf583, reinterpret_tensor(primals_335, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf584)
        del primals_336
        buf585 = buf540; del buf540  # reuse
        # Source Nodes: [q_23], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf584, buf585, 43008, 112, grid=grid(43008), stream=stream0)
        buf587 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_23], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf585, buf587, 6144, 7, grid=grid(6144), stream=stream0)
        buf588 = buf585; del buf585  # reuse
        # Source Nodes: [k_23], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf584, buf588, 43008, 112, grid=grid(43008), stream=stream0)
        buf590 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_23], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf588, buf590, 6144, 7, grid=grid(6144), stream=stream0)
        buf591 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_22, q_23], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf584, buf587, buf591, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf592 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf584, buf590, buf592, 4816896, grid=grid(4816896), stream=stream0)
        buf593 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf591, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf592, (128, 784, 48), (37632, 48, 1), 0), out=buf593)
        buf594 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf595 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf596 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_33, attn_34, attn_35], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf593, primals_46, buf594, buf595, buf596, 6144, 48, grid=grid(6144), stream=stream0)
        buf597 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf584, buf597, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf598 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf596, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf597, (128, 48, 784), (37632, 784, 1), 0), out=buf598)
        buf599 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_215], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf598, buf599, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf600 = reinterpret_tensor(buf598, (6272, 768), (768, 1), 0); del buf598  # reuse
        # Source Nodes: [x_215], Original ATen: [aten.mm]
        extern_kernels.mm(buf599, reinterpret_tensor(primals_337, (768, 768), (1, 768), 0), out=buf600)
        buf601 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf605 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf606 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1330 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_11_norm3, mul_46, mul_48, x_213, x_215, x_217, x_218], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_36.run(buf570, primals_44, buf578, primals_45, buf600, primals_338, primals_339, primals_340, buf601, buf605, buf606, buf1330, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_340
        # Source Nodes: [x_219], Original ATen: [aten.convolution]
        buf607 = extern_kernels.convolution(buf606, primals_341, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf607, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf608 = reinterpret_tensor(buf570, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf570  # reuse
        # Source Nodes: [x_219], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf607, primals_342, buf608, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_342
        buf609 = buf562; del buf562  # reuse
        buf610 = buf561; del buf561  # reuse
        buf611 = buf560; del buf560  # reuse
        # Source Nodes: [x_220, x_221], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf608, buf609, buf610, buf611, 37632, 128, grid=grid(37632), stream=stream0)
        buf612 = buf564; del buf564  # reuse
        buf613 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf615 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_220, x_221], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf609, buf610, buf611, primals_671, primals_672, buf612, buf613, buf615, primals_671, primals_672, 768, 49, grid=grid(768), stream=stream0)
        del primals_671
        del primals_672
        buf616 = reinterpret_tensor(buf607, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf607  # reuse
        # Source Nodes: [x_220, x_221], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf608, buf612, buf613, primals_343, primals_344, buf616, 4816896, grid=grid(4816896), stream=stream0)
        del primals_344
        # Source Nodes: [x_222], Original ATen: [aten.convolution]
        buf617 = extern_kernels.convolution(buf616, primals_345, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf617, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf618 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_222], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf617, primals_346, buf618, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_346
        buf622 = reinterpret_tensor(buf617, (8, 784, 768), (602112, 768, 1), 0); del buf617  # reuse
        buf623 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1329 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_11_norm2, mul_49, x_224, x_225], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf601, primals_47, buf618, primals_347, primals_348, buf622, buf623, buf1329, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_348
        buf624 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_225], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_350, buf623, reinterpret_tensor(primals_349, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf624)
        del primals_350
        buf625 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_226, x_229], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf624, buf625, 19267584, grid=grid(19267584), stream=stream0)
        buf626 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_229], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_352, buf625, reinterpret_tensor(primals_351, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf626)
        del primals_352
        buf627 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf631 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf632 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1328 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_12_attn_qkv, l__mod___blocks_12_norm1, mul_49, mul_50, x_224, x_232], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_37.run(buf601, primals_47, buf618, primals_48, buf626, primals_353, primals_354, buf627, buf631, buf632, buf1328, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_354
        buf633 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_12_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_356, buf632, reinterpret_tensor(primals_355, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf633)
        del primals_356
        buf634 = buf588; del buf588  # reuse
        # Source Nodes: [q_25], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf633, buf634, 43008, 112, grid=grid(43008), stream=stream0)
        buf636 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_25], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf634, buf636, 6144, 7, grid=grid(6144), stream=stream0)
        buf637 = buf634; del buf634  # reuse
        # Source Nodes: [k_25], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf633, buf637, 43008, 112, grid=grid(43008), stream=stream0)
        buf639 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_25], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf637, buf639, 6144, 7, grid=grid(6144), stream=stream0)
        buf640 = reinterpret_tensor(buf601, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf601  # reuse
        # Source Nodes: [matmul_24, q_25], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf633, buf636, buf640, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf641 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf633, buf639, buf641, 4816896, grid=grid(4816896), stream=stream0)
        buf642 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf640, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf641, (128, 784, 48), (37632, 48, 1), 0), out=buf642)
        buf643 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf644 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf645 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_36, attn_37, attn_38], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf642, primals_50, buf643, buf644, buf645, 6144, 48, grid=grid(6144), stream=stream0)
        buf646 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf633, buf646, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf647 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf645, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf646, (128, 48, 784), (37632, 784, 1), 0), out=buf647)
        buf648 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_234], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf647, buf648, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf649 = reinterpret_tensor(buf647, (6272, 768), (768, 1), 0); del buf647  # reuse
        # Source Nodes: [x_234], Original ATen: [aten.mm]
        extern_kernels.mm(buf648, reinterpret_tensor(primals_357, (768, 768), (1, 768), 0), out=buf649)
        buf653 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf654 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1326 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_12_norm3, mul_52, x_234, x_236, x_237], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_29.run(buf627, primals_49, buf649, primals_358, primals_359, primals_360, buf653, buf654, buf1326, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_360
        # Source Nodes: [x_238], Original ATen: [aten.convolution]
        buf655 = extern_kernels.convolution(buf654, primals_361, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf655, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf656 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_238], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf655, primals_362, buf656, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_362
        buf657 = buf611; del buf611  # reuse
        buf658 = buf610; del buf610  # reuse
        buf659 = buf609; del buf609  # reuse
        # Source Nodes: [x_239, x_240], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf656, buf657, buf658, buf659, 37632, 128, grid=grid(37632), stream=stream0)
        buf660 = buf613; del buf613  # reuse
        buf661 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf663 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_239, x_240], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf657, buf658, buf659, primals_674, primals_675, buf660, buf661, buf663, primals_674, primals_675, 768, 49, grid=grid(768), stream=stream0)
        del primals_674
        del primals_675
        buf664 = reinterpret_tensor(buf655, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf655  # reuse
        # Source Nodes: [x_239, x_240], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf656, buf660, buf661, primals_363, primals_364, buf664, 4816896, grid=grid(4816896), stream=stream0)
        del primals_364
        # Source Nodes: [x_241], Original ATen: [aten.convolution]
        buf665 = extern_kernels.convolution(buf664, primals_365, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf665, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf666 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_241], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf665, primals_366, buf666, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_366
        buf667 = reinterpret_tensor(buf665, (8, 784, 768), (602112, 768, 1), 0); del buf665  # reuse
        buf671 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf672 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1325 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_12_norm2, mul_52, mul_53, x_234, x_236, x_243, x_244], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_33.run(buf627, primals_49, buf649, primals_358, primals_51, buf666, primals_367, primals_368, buf667, buf671, buf672, buf1325, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_368
        buf673 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_244], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_370, buf672, reinterpret_tensor(primals_369, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf673)
        del primals_370
        buf674 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_245, x_248], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf673, buf674, 19267584, grid=grid(19267584), stream=stream0)
        buf675 = reinterpret_tensor(buf627, (6272, 768), (768, 1), 0); del buf627  # reuse
        # Source Nodes: [x_248], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_372, buf674, reinterpret_tensor(primals_371, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf675)
        del primals_372
        buf679 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf680 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1324 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_13_attn_qkv, l__mod___blocks_13_norm1, mul_54, x_251], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf667, primals_52, buf675, primals_373, primals_374, buf679, buf680, buf1324, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_374
        buf681 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_13_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_376, buf680, reinterpret_tensor(primals_375, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf681)
        del primals_376
        buf682 = buf637; del buf637  # reuse
        # Source Nodes: [q_27], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf681, buf682, 43008, 112, grid=grid(43008), stream=stream0)
        buf684 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_27], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf682, buf684, 6144, 7, grid=grid(6144), stream=stream0)
        buf685 = buf682; del buf682  # reuse
        # Source Nodes: [k_27], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf681, buf685, 43008, 112, grid=grid(43008), stream=stream0)
        buf687 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_27], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf685, buf687, 6144, 7, grid=grid(6144), stream=stream0)
        buf688 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_26, q_27], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf681, buf684, buf688, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf689 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf681, buf687, buf689, 4816896, grid=grid(4816896), stream=stream0)
        buf690 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf688, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf689, (128, 784, 48), (37632, 48, 1), 0), out=buf690)
        buf691 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf692 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf693 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_39, attn_40, attn_41], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf690, primals_54, buf691, buf692, buf693, 6144, 48, grid=grid(6144), stream=stream0)
        buf694 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf681, buf694, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf695 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf693, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf694, (128, 48, 784), (37632, 784, 1), 0), out=buf695)
        buf696 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_253], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf695, buf696, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf697 = reinterpret_tensor(buf695, (6272, 768), (768, 1), 0); del buf695  # reuse
        # Source Nodes: [x_253], Original ATen: [aten.mm]
        extern_kernels.mm(buf696, reinterpret_tensor(primals_377, (768, 768), (1, 768), 0), out=buf697)
        buf698 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf702 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf703 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1322 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_13_norm3, mul_54, mul_56, x_251, x_253, x_255, x_256], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_36.run(buf667, primals_52, buf675, primals_53, buf697, primals_378, primals_379, primals_380, buf698, buf702, buf703, buf1322, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_380
        # Source Nodes: [x_257], Original ATen: [aten.convolution]
        buf704 = extern_kernels.convolution(buf703, primals_381, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf704, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf705 = reinterpret_tensor(buf667, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf667  # reuse
        # Source Nodes: [x_257], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf704, primals_382, buf705, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_382
        buf706 = buf659; del buf659  # reuse
        buf707 = buf658; del buf658  # reuse
        buf708 = buf657; del buf657  # reuse
        # Source Nodes: [x_258, x_259], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf705, buf706, buf707, buf708, 37632, 128, grid=grid(37632), stream=stream0)
        buf709 = buf661; del buf661  # reuse
        buf710 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf712 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_258, x_259], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf706, buf707, buf708, primals_677, primals_678, buf709, buf710, buf712, primals_677, primals_678, 768, 49, grid=grid(768), stream=stream0)
        del primals_677
        del primals_678
        buf713 = reinterpret_tensor(buf704, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf704  # reuse
        # Source Nodes: [x_258, x_259], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf705, buf709, buf710, primals_383, primals_384, buf713, 4816896, grid=grid(4816896), stream=stream0)
        del primals_384
        # Source Nodes: [x_260], Original ATen: [aten.convolution]
        buf714 = extern_kernels.convolution(buf713, primals_385, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf714, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf715 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_260], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf714, primals_386, buf715, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_386
        buf719 = reinterpret_tensor(buf714, (8, 784, 768), (602112, 768, 1), 0); del buf714  # reuse
        buf720 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1321 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_13_norm2, mul_57, x_262, x_263], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf698, primals_55, buf715, primals_387, primals_388, buf719, buf720, buf1321, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_388
        buf721 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_263], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_390, buf720, reinterpret_tensor(primals_389, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf721)
        del primals_390
        buf722 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_264, x_267], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf721, buf722, 19267584, grid=grid(19267584), stream=stream0)
        buf723 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_267], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_392, buf722, reinterpret_tensor(primals_391, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf723)
        del primals_392
        buf724 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf728 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf729 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1320 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_14_attn_qkv, l__mod___blocks_14_norm1, mul_57, mul_58, x_262, x_270], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_37.run(buf698, primals_55, buf715, primals_56, buf723, primals_393, primals_394, buf724, buf728, buf729, buf1320, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_394
        buf730 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_14_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_396, buf729, reinterpret_tensor(primals_395, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf730)
        del primals_396
        buf731 = buf685; del buf685  # reuse
        # Source Nodes: [q_29], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf730, buf731, 43008, 112, grid=grid(43008), stream=stream0)
        buf733 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_29], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf731, buf733, 6144, 7, grid=grid(6144), stream=stream0)
        buf734 = buf731; del buf731  # reuse
        # Source Nodes: [k_29], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf730, buf734, 43008, 112, grid=grid(43008), stream=stream0)
        buf736 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_29], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf734, buf736, 6144, 7, grid=grid(6144), stream=stream0)
        buf737 = reinterpret_tensor(buf698, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf698  # reuse
        # Source Nodes: [matmul_28, q_29], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf730, buf733, buf737, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf738 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf730, buf736, buf738, 4816896, grid=grid(4816896), stream=stream0)
        buf739 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf737, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf738, (128, 784, 48), (37632, 48, 1), 0), out=buf739)
        buf740 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf741 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf742 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_42, attn_43, attn_44], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf739, primals_58, buf740, buf741, buf742, 6144, 48, grid=grid(6144), stream=stream0)
        buf743 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf730, buf743, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf744 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf742, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf743, (128, 48, 784), (37632, 784, 1), 0), out=buf744)
        buf745 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_272], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf744, buf745, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf746 = reinterpret_tensor(buf744, (6272, 768), (768, 1), 0); del buf744  # reuse
        # Source Nodes: [x_272], Original ATen: [aten.mm]
        extern_kernels.mm(buf745, reinterpret_tensor(primals_397, (768, 768), (1, 768), 0), out=buf746)
        buf750 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf751 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1318 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_14_norm3, mul_60, x_272, x_274, x_275], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_29.run(buf724, primals_57, buf746, primals_398, primals_399, primals_400, buf750, buf751, buf1318, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_400
        # Source Nodes: [x_276], Original ATen: [aten.convolution]
        buf752 = extern_kernels.convolution(buf751, primals_401, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf752, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf753 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_276], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf752, primals_402, buf753, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_402
        buf754 = buf708; del buf708  # reuse
        buf755 = buf707; del buf707  # reuse
        buf756 = buf706; del buf706  # reuse
        # Source Nodes: [x_277, x_278], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf753, buf754, buf755, buf756, 37632, 128, grid=grid(37632), stream=stream0)
        buf757 = buf710; del buf710  # reuse
        buf758 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf760 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_277, x_278], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf754, buf755, buf756, primals_680, primals_681, buf757, buf758, buf760, primals_680, primals_681, 768, 49, grid=grid(768), stream=stream0)
        del primals_680
        del primals_681
        buf761 = reinterpret_tensor(buf752, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf752  # reuse
        # Source Nodes: [x_277, x_278], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf753, buf757, buf758, primals_403, primals_404, buf761, 4816896, grid=grid(4816896), stream=stream0)
        del primals_404
        # Source Nodes: [x_279], Original ATen: [aten.convolution]
        buf762 = extern_kernels.convolution(buf761, primals_405, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf762, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf763 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_279], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf762, primals_406, buf763, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_406
        buf764 = reinterpret_tensor(buf762, (8, 784, 768), (602112, 768, 1), 0); del buf762  # reuse
        buf768 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf769 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1317 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_14_norm2, mul_60, mul_61, x_272, x_274, x_281, x_282], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_33.run(buf724, primals_57, buf746, primals_398, primals_59, buf763, primals_407, primals_408, buf764, buf768, buf769, buf1317, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_408
        buf770 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_282], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_410, buf769, reinterpret_tensor(primals_409, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf770)
        del primals_410
        buf771 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_283, x_286], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf770, buf771, 19267584, grid=grid(19267584), stream=stream0)
        buf772 = reinterpret_tensor(buf724, (6272, 768), (768, 1), 0); del buf724  # reuse
        # Source Nodes: [x_286], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_412, buf771, reinterpret_tensor(primals_411, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf772)
        del primals_412
        buf776 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf777 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1316 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_15_attn_qkv, l__mod___blocks_15_norm1, mul_62, x_289], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf764, primals_60, buf772, primals_413, primals_414, buf776, buf777, buf1316, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_414
        buf778 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_15_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_416, buf777, reinterpret_tensor(primals_415, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf778)
        del primals_416
        buf779 = buf734; del buf734  # reuse
        # Source Nodes: [q_31], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf778, buf779, 43008, 112, grid=grid(43008), stream=stream0)
        buf781 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_31], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf779, buf781, 6144, 7, grid=grid(6144), stream=stream0)
        buf782 = buf779; del buf779  # reuse
        # Source Nodes: [k_31], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf778, buf782, 43008, 112, grid=grid(43008), stream=stream0)
        buf784 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_31], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf782, buf784, 6144, 7, grid=grid(6144), stream=stream0)
        buf785 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_30, q_31], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf778, buf781, buf785, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf786 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf778, buf784, buf786, 4816896, grid=grid(4816896), stream=stream0)
        buf787 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf785, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf786, (128, 784, 48), (37632, 48, 1), 0), out=buf787)
        buf788 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf789 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf790 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_45, attn_46, attn_47], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf787, primals_62, buf788, buf789, buf790, 6144, 48, grid=grid(6144), stream=stream0)
        buf791 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf778, buf791, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf792 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf790, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf791, (128, 48, 784), (37632, 784, 1), 0), out=buf792)
        buf793 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_291], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf792, buf793, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf794 = reinterpret_tensor(buf792, (6272, 768), (768, 1), 0); del buf792  # reuse
        # Source Nodes: [x_291], Original ATen: [aten.mm]
        extern_kernels.mm(buf793, reinterpret_tensor(primals_417, (768, 768), (1, 768), 0), out=buf794)
        buf795 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf799 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf800 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1314 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_15_norm3, mul_62, mul_64, x_289, x_291, x_293, x_294], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_36.run(buf764, primals_60, buf772, primals_61, buf794, primals_418, primals_419, primals_420, buf795, buf799, buf800, buf1314, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_420
        # Source Nodes: [x_295], Original ATen: [aten.convolution]
        buf801 = extern_kernels.convolution(buf800, primals_421, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf801, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf802 = reinterpret_tensor(buf764, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf764  # reuse
        # Source Nodes: [x_295], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf801, primals_422, buf802, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_422
        buf803 = buf756; del buf756  # reuse
        buf804 = buf755; del buf755  # reuse
        buf805 = buf754; del buf754  # reuse
        # Source Nodes: [x_296, x_297], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf802, buf803, buf804, buf805, 37632, 128, grid=grid(37632), stream=stream0)
        buf806 = buf758; del buf758  # reuse
        buf807 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf809 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_296, x_297], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf803, buf804, buf805, primals_683, primals_684, buf806, buf807, buf809, primals_683, primals_684, 768, 49, grid=grid(768), stream=stream0)
        del primals_683
        del primals_684
        buf810 = reinterpret_tensor(buf801, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf801  # reuse
        # Source Nodes: [x_296, x_297], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf802, buf806, buf807, primals_423, primals_424, buf810, 4816896, grid=grid(4816896), stream=stream0)
        del primals_424
        # Source Nodes: [x_298], Original ATen: [aten.convolution]
        buf811 = extern_kernels.convolution(buf810, primals_425, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf811, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf812 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_298], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf811, primals_426, buf812, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_426
        buf816 = reinterpret_tensor(buf811, (8, 784, 768), (602112, 768, 1), 0); del buf811  # reuse
        buf817 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1313 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_15_norm2, mul_65, x_300, x_301], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf795, primals_63, buf812, primals_427, primals_428, buf816, buf817, buf1313, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_428
        buf818 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_301], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_430, buf817, reinterpret_tensor(primals_429, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf818)
        del primals_430
        buf819 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_302, x_305], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf818, buf819, 19267584, grid=grid(19267584), stream=stream0)
        buf820 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_305], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_432, buf819, reinterpret_tensor(primals_431, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf820)
        del primals_432
        buf821 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf825 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf826 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1312 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_16_attn_qkv, l__mod___blocks_16_norm1, mul_65, mul_66, x_300, x_308], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_37.run(buf795, primals_63, buf812, primals_64, buf820, primals_433, primals_434, buf821, buf825, buf826, buf1312, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_434
        buf827 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_16_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_436, buf826, reinterpret_tensor(primals_435, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf827)
        del primals_436
        buf828 = buf782; del buf782  # reuse
        # Source Nodes: [q_33], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf827, buf828, 43008, 112, grid=grid(43008), stream=stream0)
        buf830 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_33], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf828, buf830, 6144, 7, grid=grid(6144), stream=stream0)
        buf831 = buf828; del buf828  # reuse
        # Source Nodes: [k_33], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf827, buf831, 43008, 112, grid=grid(43008), stream=stream0)
        buf833 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_33], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf831, buf833, 6144, 7, grid=grid(6144), stream=stream0)
        buf834 = reinterpret_tensor(buf795, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf795  # reuse
        # Source Nodes: [matmul_32, q_33], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf827, buf830, buf834, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf835 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf827, buf833, buf835, 4816896, grid=grid(4816896), stream=stream0)
        buf836 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf834, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf835, (128, 784, 48), (37632, 48, 1), 0), out=buf836)
        buf837 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf838 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf839 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_48, attn_49, attn_50], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf836, primals_66, buf837, buf838, buf839, 6144, 48, grid=grid(6144), stream=stream0)
        buf840 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf827, buf840, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf841 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf839, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf840, (128, 48, 784), (37632, 784, 1), 0), out=buf841)
        buf842 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_310], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf841, buf842, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf843 = reinterpret_tensor(buf841, (6272, 768), (768, 1), 0); del buf841  # reuse
        # Source Nodes: [x_310], Original ATen: [aten.mm]
        extern_kernels.mm(buf842, reinterpret_tensor(primals_437, (768, 768), (1, 768), 0), out=buf843)
        buf847 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf848 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1310 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_16_norm3, mul_68, x_310, x_312, x_313], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_29.run(buf821, primals_65, buf843, primals_438, primals_439, primals_440, buf847, buf848, buf1310, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_440
        # Source Nodes: [x_314], Original ATen: [aten.convolution]
        buf849 = extern_kernels.convolution(buf848, primals_441, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf849, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf850 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_314], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf849, primals_442, buf850, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_442
        buf851 = buf805; del buf805  # reuse
        buf852 = buf804; del buf804  # reuse
        buf853 = buf803; del buf803  # reuse
        # Source Nodes: [x_315, x_316], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf850, buf851, buf852, buf853, 37632, 128, grid=grid(37632), stream=stream0)
        buf854 = buf807; del buf807  # reuse
        buf855 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf857 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_315, x_316], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf851, buf852, buf853, primals_686, primals_687, buf854, buf855, buf857, primals_686, primals_687, 768, 49, grid=grid(768), stream=stream0)
        del primals_686
        del primals_687
        buf858 = reinterpret_tensor(buf849, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf849  # reuse
        # Source Nodes: [x_315, x_316], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf850, buf854, buf855, primals_443, primals_444, buf858, 4816896, grid=grid(4816896), stream=stream0)
        del primals_444
        # Source Nodes: [x_317], Original ATen: [aten.convolution]
        buf859 = extern_kernels.convolution(buf858, primals_445, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf859, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf860 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_317], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf859, primals_446, buf860, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_446
        buf861 = reinterpret_tensor(buf859, (8, 784, 768), (602112, 768, 1), 0); del buf859  # reuse
        buf865 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf866 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1309 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_16_norm2, mul_68, mul_69, x_310, x_312, x_319, x_320], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_33.run(buf821, primals_65, buf843, primals_438, primals_67, buf860, primals_447, primals_448, buf861, buf865, buf866, buf1309, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_448
        buf867 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_320], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_450, buf866, reinterpret_tensor(primals_449, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf867)
        del primals_450
        buf868 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_321, x_324], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf867, buf868, 19267584, grid=grid(19267584), stream=stream0)
        buf869 = reinterpret_tensor(buf821, (6272, 768), (768, 1), 0); del buf821  # reuse
        # Source Nodes: [x_324], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_452, buf868, reinterpret_tensor(primals_451, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf869)
        del primals_452
        buf873 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf874 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1308 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_17_attn_qkv, l__mod___blocks_17_norm1, mul_70, x_327], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf861, primals_68, buf869, primals_453, primals_454, buf873, buf874, buf1308, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_454
        buf875 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_17_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_456, buf874, reinterpret_tensor(primals_455, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf875)
        del primals_456
        buf876 = buf831; del buf831  # reuse
        # Source Nodes: [q_35], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf875, buf876, 43008, 112, grid=grid(43008), stream=stream0)
        buf878 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_35], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf876, buf878, 6144, 7, grid=grid(6144), stream=stream0)
        buf879 = buf876; del buf876  # reuse
        # Source Nodes: [k_35], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf875, buf879, 43008, 112, grid=grid(43008), stream=stream0)
        buf881 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_35], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf879, buf881, 6144, 7, grid=grid(6144), stream=stream0)
        buf882 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_34, q_35], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf875, buf878, buf882, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf883 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf875, buf881, buf883, 4816896, grid=grid(4816896), stream=stream0)
        buf884 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf882, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf883, (128, 784, 48), (37632, 48, 1), 0), out=buf884)
        buf885 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf886 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf887 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_51, attn_52, attn_53], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf884, primals_70, buf885, buf886, buf887, 6144, 48, grid=grid(6144), stream=stream0)
        buf888 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf875, buf888, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf889 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf887, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf888, (128, 48, 784), (37632, 784, 1), 0), out=buf889)
        buf890 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_329], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf889, buf890, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf891 = reinterpret_tensor(buf889, (6272, 768), (768, 1), 0); del buf889  # reuse
        # Source Nodes: [x_329], Original ATen: [aten.mm]
        extern_kernels.mm(buf890, reinterpret_tensor(primals_457, (768, 768), (1, 768), 0), out=buf891)
        buf892 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf896 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf897 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1306 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_17_norm3, mul_70, mul_72, x_327, x_329, x_331, x_332], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_36.run(buf861, primals_68, buf869, primals_69, buf891, primals_458, primals_459, primals_460, buf892, buf896, buf897, buf1306, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_460
        # Source Nodes: [x_333], Original ATen: [aten.convolution]
        buf898 = extern_kernels.convolution(buf897, primals_461, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf898, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf899 = reinterpret_tensor(buf861, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf861  # reuse
        # Source Nodes: [x_333], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf898, primals_462, buf899, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_462
        buf900 = buf853; del buf853  # reuse
        buf901 = buf852; del buf852  # reuse
        buf902 = buf851; del buf851  # reuse
        # Source Nodes: [x_334, x_335], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf899, buf900, buf901, buf902, 37632, 128, grid=grid(37632), stream=stream0)
        buf903 = buf855; del buf855  # reuse
        buf904 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf906 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_334, x_335], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf900, buf901, buf902, primals_689, primals_690, buf903, buf904, buf906, primals_689, primals_690, 768, 49, grid=grid(768), stream=stream0)
        del primals_689
        del primals_690
        buf907 = reinterpret_tensor(buf898, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf898  # reuse
        # Source Nodes: [x_334, x_335], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf899, buf903, buf904, primals_463, primals_464, buf907, 4816896, grid=grid(4816896), stream=stream0)
        del primals_464
        # Source Nodes: [x_336], Original ATen: [aten.convolution]
        buf908 = extern_kernels.convolution(buf907, primals_465, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf908, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf909 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_336], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf908, primals_466, buf909, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_466
        buf913 = reinterpret_tensor(buf908, (8, 784, 768), (602112, 768, 1), 0); del buf908  # reuse
        buf914 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1305 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_17_norm2, mul_73, x_338, x_339], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf892, primals_71, buf909, primals_467, primals_468, buf913, buf914, buf1305, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_468
        buf915 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_339], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_470, buf914, reinterpret_tensor(primals_469, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf915)
        del primals_470
        buf916 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_340, x_343], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf915, buf916, 19267584, grid=grid(19267584), stream=stream0)
        buf917 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_343], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_472, buf916, reinterpret_tensor(primals_471, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf917)
        del primals_472
        buf918 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf922 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf923 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1304 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_18_attn_qkv, l__mod___blocks_18_norm1, mul_73, mul_74, x_338, x_346], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_37.run(buf892, primals_71, buf909, primals_72, buf917, primals_473, primals_474, buf918, buf922, buf923, buf1304, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_474
        buf924 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_18_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_476, buf923, reinterpret_tensor(primals_475, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf924)
        del primals_476
        buf925 = buf879; del buf879  # reuse
        # Source Nodes: [q_37], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf924, buf925, 43008, 112, grid=grid(43008), stream=stream0)
        buf927 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_37], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf925, buf927, 6144, 7, grid=grid(6144), stream=stream0)
        buf928 = buf925; del buf925  # reuse
        # Source Nodes: [k_37], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf924, buf928, 43008, 112, grid=grid(43008), stream=stream0)
        buf930 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_37], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf928, buf930, 6144, 7, grid=grid(6144), stream=stream0)
        buf931 = reinterpret_tensor(buf892, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf892  # reuse
        # Source Nodes: [matmul_36, q_37], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf924, buf927, buf931, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf932 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf924, buf930, buf932, 4816896, grid=grid(4816896), stream=stream0)
        buf933 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf931, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf932, (128, 784, 48), (37632, 48, 1), 0), out=buf933)
        buf934 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf935 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf936 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_54, attn_55, attn_56], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf933, primals_74, buf934, buf935, buf936, 6144, 48, grid=grid(6144), stream=stream0)
        buf937 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_37], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf924, buf937, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf938 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_37], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf936, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf937, (128, 48, 784), (37632, 784, 1), 0), out=buf938)
        buf939 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_348], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf938, buf939, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf940 = reinterpret_tensor(buf938, (6272, 768), (768, 1), 0); del buf938  # reuse
        # Source Nodes: [x_348], Original ATen: [aten.mm]
        extern_kernels.mm(buf939, reinterpret_tensor(primals_477, (768, 768), (1, 768), 0), out=buf940)
        buf944 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf945 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1302 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_18_norm3, mul_76, x_348, x_350, x_351], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_29.run(buf918, primals_73, buf940, primals_478, primals_479, primals_480, buf944, buf945, buf1302, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_480
        # Source Nodes: [x_352], Original ATen: [aten.convolution]
        buf946 = extern_kernels.convolution(buf945, primals_481, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf946, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf947 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_352], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf946, primals_482, buf947, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_482
        buf948 = buf902; del buf902  # reuse
        buf949 = buf901; del buf901  # reuse
        buf950 = buf900; del buf900  # reuse
        # Source Nodes: [x_353, x_354], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf947, buf948, buf949, buf950, 37632, 128, grid=grid(37632), stream=stream0)
        buf951 = buf904; del buf904  # reuse
        buf952 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf954 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_353, x_354], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf948, buf949, buf950, primals_692, primals_693, buf951, buf952, buf954, primals_692, primals_693, 768, 49, grid=grid(768), stream=stream0)
        del primals_692
        del primals_693
        buf955 = reinterpret_tensor(buf946, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf946  # reuse
        # Source Nodes: [x_353, x_354], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf947, buf951, buf952, primals_483, primals_484, buf955, 4816896, grid=grid(4816896), stream=stream0)
        del primals_484
        # Source Nodes: [x_355], Original ATen: [aten.convolution]
        buf956 = extern_kernels.convolution(buf955, primals_485, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf956, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf957 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_355], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf956, primals_486, buf957, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_486
        buf958 = reinterpret_tensor(buf956, (8, 784, 768), (602112, 768, 1), 0); del buf956  # reuse
        buf962 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf963 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1301 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_18_norm2, mul_76, mul_77, x_348, x_350, x_357, x_358], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_33.run(buf918, primals_73, buf940, primals_478, primals_75, buf957, primals_487, primals_488, buf958, buf962, buf963, buf1301, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_488
        buf964 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_358], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_490, buf963, reinterpret_tensor(primals_489, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf964)
        del primals_490
        buf965 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_359, x_362], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf964, buf965, 19267584, grid=grid(19267584), stream=stream0)
        buf966 = reinterpret_tensor(buf918, (6272, 768), (768, 1), 0); del buf918  # reuse
        # Source Nodes: [x_362], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_492, buf965, reinterpret_tensor(primals_491, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf966)
        del primals_492
        buf970 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf971 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1300 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_19_attn_qkv, l__mod___blocks_19_norm1, mul_78, x_365], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf958, primals_76, buf966, primals_493, primals_494, buf970, buf971, buf1300, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_494
        buf972 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_19_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_496, buf971, reinterpret_tensor(primals_495, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf972)
        del primals_496
        buf973 = buf928; del buf928  # reuse
        # Source Nodes: [q_39], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf972, buf973, 43008, 112, grid=grid(43008), stream=stream0)
        buf975 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_39], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf973, buf975, 6144, 7, grid=grid(6144), stream=stream0)
        buf976 = buf973; del buf973  # reuse
        # Source Nodes: [k_39], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf972, buf976, 43008, 112, grid=grid(43008), stream=stream0)
        buf978 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_39], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf976, buf978, 6144, 7, grid=grid(6144), stream=stream0)
        buf979 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_38, q_39], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf972, buf975, buf979, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf980 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf972, buf978, buf980, 4816896, grid=grid(4816896), stream=stream0)
        buf981 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_38], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf979, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf980, (128, 784, 48), (37632, 48, 1), 0), out=buf981)
        buf982 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf983 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf984 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_57, attn_58, attn_59], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf981, primals_78, buf982, buf983, buf984, 6144, 48, grid=grid(6144), stream=stream0)
        buf985 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_39], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf972, buf985, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf986 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_39], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf984, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf985, (128, 48, 784), (37632, 784, 1), 0), out=buf986)
        buf987 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_367], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf986, buf987, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf988 = reinterpret_tensor(buf986, (6272, 768), (768, 1), 0); del buf986  # reuse
        # Source Nodes: [x_367], Original ATen: [aten.mm]
        extern_kernels.mm(buf987, reinterpret_tensor(primals_497, (768, 768), (1, 768), 0), out=buf988)
        buf989 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf993 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf994 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1298 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_19_norm3, mul_78, mul_80, x_365, x_367, x_369, x_370], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_36.run(buf958, primals_76, buf966, primals_77, buf988, primals_498, primals_499, primals_500, buf989, buf993, buf994, buf1298, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_500
        # Source Nodes: [x_371], Original ATen: [aten.convolution]
        buf995 = extern_kernels.convolution(buf994, primals_501, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf995, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf996 = reinterpret_tensor(buf958, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf958  # reuse
        # Source Nodes: [x_371], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf995, primals_502, buf996, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_502
        buf997 = buf950; del buf950  # reuse
        buf998 = buf949; del buf949  # reuse
        buf999 = buf948; del buf948  # reuse
        # Source Nodes: [x_372, x_373], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf996, buf997, buf998, buf999, 37632, 128, grid=grid(37632), stream=stream0)
        buf1000 = buf952; del buf952  # reuse
        buf1001 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf1003 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_372, x_373], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf997, buf998, buf999, primals_695, primals_696, buf1000, buf1001, buf1003, primals_695, primals_696, 768, 49, grid=grid(768), stream=stream0)
        del primals_695
        del primals_696
        buf1004 = reinterpret_tensor(buf995, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf995  # reuse
        # Source Nodes: [x_372, x_373], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf996, buf1000, buf1001, primals_503, primals_504, buf1004, 4816896, grid=grid(4816896), stream=stream0)
        del primals_504
        # Source Nodes: [x_374], Original ATen: [aten.convolution]
        buf1005 = extern_kernels.convolution(buf1004, primals_505, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf1005, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf1006 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_374], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf1005, primals_506, buf1006, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_506
        buf1010 = reinterpret_tensor(buf1005, (8, 784, 768), (602112, 768, 1), 0); del buf1005  # reuse
        buf1011 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1297 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_19_norm2, mul_81, x_376, x_377], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf989, primals_79, buf1006, primals_507, primals_508, buf1010, buf1011, buf1297, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_508
        buf1012 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_377], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_510, buf1011, reinterpret_tensor(primals_509, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf1012)
        del primals_510
        buf1013 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_378, x_381], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf1012, buf1013, 19267584, grid=grid(19267584), stream=stream0)
        buf1014 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_381], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_512, buf1013, reinterpret_tensor(primals_511, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf1014)
        del primals_512
        buf1015 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf1019 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf1020 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1296 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_20_attn_qkv, l__mod___blocks_20_norm1, mul_81, mul_82, x_376, x_384], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_37.run(buf989, primals_79, buf1006, primals_80, buf1014, primals_513, primals_514, buf1015, buf1019, buf1020, buf1296, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_514
        buf1021 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_20_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_516, buf1020, reinterpret_tensor(primals_515, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf1021)
        del primals_516
        buf1022 = buf976; del buf976  # reuse
        # Source Nodes: [q_41], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf1021, buf1022, 43008, 112, grid=grid(43008), stream=stream0)
        buf1024 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_41], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf1022, buf1024, 6144, 7, grid=grid(6144), stream=stream0)
        buf1025 = buf1022; del buf1022  # reuse
        # Source Nodes: [k_41], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf1021, buf1025, 43008, 112, grid=grid(43008), stream=stream0)
        buf1027 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_41], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf1025, buf1027, 6144, 7, grid=grid(6144), stream=stream0)
        buf1028 = reinterpret_tensor(buf989, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf989  # reuse
        # Source Nodes: [matmul_40, q_41], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf1021, buf1024, buf1028, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1029 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf1021, buf1027, buf1029, 4816896, grid=grid(4816896), stream=stream0)
        buf1030 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1028, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf1029, (128, 784, 48), (37632, 48, 1), 0), out=buf1030)
        buf1031 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf1032 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf1033 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_60, attn_61, attn_62], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf1030, primals_82, buf1031, buf1032, buf1033, 6144, 48, grid=grid(6144), stream=stream0)
        buf1034 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_41], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf1021, buf1034, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1035 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_41], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1033, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf1034, (128, 48, 784), (37632, 784, 1), 0), out=buf1035)
        buf1036 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_386], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf1035, buf1036, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf1037 = reinterpret_tensor(buf1035, (6272, 768), (768, 1), 0); del buf1035  # reuse
        # Source Nodes: [x_386], Original ATen: [aten.mm]
        extern_kernels.mm(buf1036, reinterpret_tensor(primals_517, (768, 768), (1, 768), 0), out=buf1037)
        buf1041 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf1042 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1294 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_20_norm3, mul_84, x_386, x_388, x_389], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_29.run(buf1015, primals_81, buf1037, primals_518, primals_519, primals_520, buf1041, buf1042, buf1294, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_520
        # Source Nodes: [x_390], Original ATen: [aten.convolution]
        buf1043 = extern_kernels.convolution(buf1042, primals_521, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf1043, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf1044 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_390], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf1043, primals_522, buf1044, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_522
        buf1045 = buf999; del buf999  # reuse
        buf1046 = buf998; del buf998  # reuse
        buf1047 = buf997; del buf997  # reuse
        # Source Nodes: [x_391, x_392], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf1044, buf1045, buf1046, buf1047, 37632, 128, grid=grid(37632), stream=stream0)
        buf1048 = buf1001; del buf1001  # reuse
        buf1049 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf1051 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_391, x_392], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf1045, buf1046, buf1047, primals_698, primals_699, buf1048, buf1049, buf1051, primals_698, primals_699, 768, 49, grid=grid(768), stream=stream0)
        del primals_698
        del primals_699
        buf1052 = reinterpret_tensor(buf1043, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf1043  # reuse
        # Source Nodes: [x_391, x_392], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf1044, buf1048, buf1049, primals_523, primals_524, buf1052, 4816896, grid=grid(4816896), stream=stream0)
        del primals_524
        # Source Nodes: [x_393], Original ATen: [aten.convolution]
        buf1053 = extern_kernels.convolution(buf1052, primals_525, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf1053, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf1054 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_393], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf1053, primals_526, buf1054, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_526
        buf1055 = reinterpret_tensor(buf1053, (8, 784, 768), (602112, 768, 1), 0); del buf1053  # reuse
        buf1059 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf1060 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1293 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_20_norm2, mul_84, mul_85, x_386, x_388, x_395, x_396], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_33.run(buf1015, primals_81, buf1037, primals_518, primals_83, buf1054, primals_527, primals_528, buf1055, buf1059, buf1060, buf1293, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_528
        buf1061 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_396], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_530, buf1060, reinterpret_tensor(primals_529, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf1061)
        del primals_530
        buf1062 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_397, x_400], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf1061, buf1062, 19267584, grid=grid(19267584), stream=stream0)
        buf1063 = reinterpret_tensor(buf1015, (6272, 768), (768, 1), 0); del buf1015  # reuse
        # Source Nodes: [x_400], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_532, buf1062, reinterpret_tensor(primals_531, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf1063)
        del primals_532
        buf1067 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf1068 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1292 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_21_attn_qkv, l__mod___blocks_21_norm1, mul_86, x_403], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf1055, primals_84, buf1063, primals_533, primals_534, buf1067, buf1068, buf1292, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_534
        buf1069 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_21_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_536, buf1068, reinterpret_tensor(primals_535, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf1069)
        del primals_536
        buf1070 = buf1025; del buf1025  # reuse
        # Source Nodes: [q_43], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf1069, buf1070, 43008, 112, grid=grid(43008), stream=stream0)
        buf1072 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_43], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf1070, buf1072, 6144, 7, grid=grid(6144), stream=stream0)
        buf1073 = buf1070; del buf1070  # reuse
        # Source Nodes: [k_43], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf1069, buf1073, 43008, 112, grid=grid(43008), stream=stream0)
        buf1075 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_43], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf1073, buf1075, 6144, 7, grid=grid(6144), stream=stream0)
        buf1076 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_42, q_43], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf1069, buf1072, buf1076, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1077 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf1069, buf1075, buf1077, 4816896, grid=grid(4816896), stream=stream0)
        buf1078 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1076, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf1077, (128, 784, 48), (37632, 48, 1), 0), out=buf1078)
        buf1079 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf1080 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf1081 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_63, attn_64, attn_65], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf1078, primals_86, buf1079, buf1080, buf1081, 6144, 48, grid=grid(6144), stream=stream0)
        buf1082 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf1069, buf1082, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1083 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_43], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1081, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf1082, (128, 48, 784), (37632, 784, 1), 0), out=buf1083)
        buf1084 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_405], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf1083, buf1084, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf1085 = reinterpret_tensor(buf1083, (6272, 768), (768, 1), 0); del buf1083  # reuse
        # Source Nodes: [x_405], Original ATen: [aten.mm]
        extern_kernels.mm(buf1084, reinterpret_tensor(primals_537, (768, 768), (1, 768), 0), out=buf1085)
        buf1086 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf1090 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf1091 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1290 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_21_norm3, mul_86, mul_88, x_403, x_405, x_407, x_408], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_36.run(buf1055, primals_84, buf1063, primals_85, buf1085, primals_538, primals_539, primals_540, buf1086, buf1090, buf1091, buf1290, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_540
        # Source Nodes: [x_409], Original ATen: [aten.convolution]
        buf1092 = extern_kernels.convolution(buf1091, primals_541, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf1092, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf1093 = reinterpret_tensor(buf1055, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf1055  # reuse
        # Source Nodes: [x_409], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf1092, primals_542, buf1093, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_542
        buf1094 = buf1047; del buf1047  # reuse
        buf1095 = buf1046; del buf1046  # reuse
        buf1096 = buf1045; del buf1045  # reuse
        # Source Nodes: [x_410, x_411], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf1093, buf1094, buf1095, buf1096, 37632, 128, grid=grid(37632), stream=stream0)
        buf1097 = buf1049; del buf1049  # reuse
        buf1098 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf1100 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_410, x_411], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf1094, buf1095, buf1096, primals_701, primals_702, buf1097, buf1098, buf1100, primals_701, primals_702, 768, 49, grid=grid(768), stream=stream0)
        del primals_701
        del primals_702
        buf1101 = reinterpret_tensor(buf1092, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf1092  # reuse
        # Source Nodes: [x_410, x_411], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf1093, buf1097, buf1098, primals_543, primals_544, buf1101, 4816896, grid=grid(4816896), stream=stream0)
        del primals_544
        # Source Nodes: [x_412], Original ATen: [aten.convolution]
        buf1102 = extern_kernels.convolution(buf1101, primals_545, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf1102, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf1103 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_412], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf1102, primals_546, buf1103, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_546
        buf1107 = reinterpret_tensor(buf1102, (8, 784, 768), (602112, 768, 1), 0); del buf1102  # reuse
        buf1108 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1289 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_21_norm2, mul_89, x_414, x_415], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf1086, primals_87, buf1103, primals_547, primals_548, buf1107, buf1108, buf1289, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_548
        buf1109 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_415], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_550, buf1108, reinterpret_tensor(primals_549, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf1109)
        del primals_550
        buf1110 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_416, x_419], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf1109, buf1110, 19267584, grid=grid(19267584), stream=stream0)
        buf1111 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_419], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_552, buf1110, reinterpret_tensor(primals_551, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf1111)
        del primals_552
        buf1112 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf1116 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf1117 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1288 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_22_attn_qkv, l__mod___blocks_22_norm1, mul_89, mul_90, x_414, x_422], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_37.run(buf1086, primals_87, buf1103, primals_88, buf1111, primals_553, primals_554, buf1112, buf1116, buf1117, buf1288, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_554
        buf1118 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_22_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_556, buf1117, reinterpret_tensor(primals_555, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf1118)
        del primals_556
        buf1119 = buf1073; del buf1073  # reuse
        # Source Nodes: [q_45], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf1118, buf1119, 43008, 112, grid=grid(43008), stream=stream0)
        buf1121 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_45], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf1119, buf1121, 6144, 7, grid=grid(6144), stream=stream0)
        buf1122 = buf1119; del buf1119  # reuse
        # Source Nodes: [k_45], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf1118, buf1122, 43008, 112, grid=grid(43008), stream=stream0)
        buf1124 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_45], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf1122, buf1124, 6144, 7, grid=grid(6144), stream=stream0)
        buf1125 = reinterpret_tensor(buf1086, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1086  # reuse
        # Source Nodes: [matmul_44, q_45], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf1118, buf1121, buf1125, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1126 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf1118, buf1124, buf1126, 4816896, grid=grid(4816896), stream=stream0)
        buf1127 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1125, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf1126, (128, 784, 48), (37632, 48, 1), 0), out=buf1127)
        buf1128 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf1129 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf1130 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_66, attn_67, attn_68], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf1127, primals_90, buf1128, buf1129, buf1130, 6144, 48, grid=grid(6144), stream=stream0)
        buf1131 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_45], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf1118, buf1131, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1132 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1130, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf1131, (128, 48, 784), (37632, 784, 1), 0), out=buf1132)
        buf1133 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_424], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf1132, buf1133, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf1134 = reinterpret_tensor(buf1132, (6272, 768), (768, 1), 0); del buf1132  # reuse
        # Source Nodes: [x_424], Original ATen: [aten.mm]
        extern_kernels.mm(buf1133, reinterpret_tensor(primals_557, (768, 768), (1, 768), 0), out=buf1134)
        buf1138 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf1139 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1286 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_22_norm3, mul_92, x_424, x_426, x_427], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_29.run(buf1112, primals_89, buf1134, primals_558, primals_559, primals_560, buf1138, buf1139, buf1286, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_560
        # Source Nodes: [x_428], Original ATen: [aten.convolution]
        buf1140 = extern_kernels.convolution(buf1139, primals_561, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf1140, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf1141 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_428], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf1140, primals_562, buf1141, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_562
        buf1142 = buf1096; del buf1096  # reuse
        buf1143 = buf1095; del buf1095  # reuse
        buf1144 = buf1094; del buf1094  # reuse
        # Source Nodes: [x_429, x_430], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf1141, buf1142, buf1143, buf1144, 37632, 128, grid=grid(37632), stream=stream0)
        buf1145 = buf1098; del buf1098  # reuse
        buf1146 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf1148 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_429, x_430], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf1142, buf1143, buf1144, primals_704, primals_705, buf1145, buf1146, buf1148, primals_704, primals_705, 768, 49, grid=grid(768), stream=stream0)
        del primals_704
        del primals_705
        buf1149 = reinterpret_tensor(buf1140, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf1140  # reuse
        # Source Nodes: [x_429, x_430], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf1141, buf1145, buf1146, primals_563, primals_564, buf1149, 4816896, grid=grid(4816896), stream=stream0)
        del primals_564
        # Source Nodes: [x_431], Original ATen: [aten.convolution]
        buf1150 = extern_kernels.convolution(buf1149, primals_565, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf1150, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf1151 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_431], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf1150, primals_566, buf1151, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_566
        buf1152 = reinterpret_tensor(buf1150, (8, 784, 768), (602112, 768, 1), 0); del buf1150  # reuse
        buf1156 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf1157 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1285 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_22_norm2, mul_92, mul_93, x_424, x_426, x_433, x_434], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_33.run(buf1112, primals_89, buf1134, primals_558, primals_91, buf1151, primals_567, primals_568, buf1152, buf1156, buf1157, buf1285, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_568
        buf1158 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_434], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_570, buf1157, reinterpret_tensor(primals_569, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf1158)
        del primals_570
        buf1159 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_435, x_438], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf1158, buf1159, 19267584, grid=grid(19267584), stream=stream0)
        buf1160 = reinterpret_tensor(buf1112, (6272, 768), (768, 1), 0); del buf1112  # reuse
        # Source Nodes: [x_438], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_572, buf1159, reinterpret_tensor(primals_571, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf1160)
        del primals_572
        buf1164 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf1165 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1284 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_23_attn_qkv, l__mod___blocks_23_norm1, mul_94, x_441], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf1152, primals_92, buf1160, primals_573, primals_574, buf1164, buf1165, buf1284, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_574
        buf1166 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_23_attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_576, buf1165, reinterpret_tensor(primals_575, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf1166)
        del primals_576
        buf1167 = buf1122; del buf1122  # reuse
        # Source Nodes: [q_47], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_21.run(buf1166, buf1167, 43008, 112, grid=grid(43008), stream=stream0)
        buf1169 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_47], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf1167, buf1169, 6144, 7, grid=grid(6144), stream=stream0)
        buf1170 = buf1167; del buf1167  # reuse
        # Source Nodes: [k_47], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_23.run(buf1166, buf1170, 43008, 112, grid=grid(43008), stream=stream0)
        buf1172 = empty_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_47], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_22.run(buf1170, buf1172, 6144, 7, grid=grid(6144), stream=stream0)
        del buf1170
        buf1173 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_46, q_47], Original ATen: [aten.clone, aten.div]
        triton_poi_fused_clone_div_24.run(buf1166, buf1169, buf1173, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1174 = empty((8, 16, 784, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf1166, buf1172, buf1174, 4816896, grid=grid(4816896), stream=stream0)
        buf1175 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1173, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf1174, (128, 784, 48), (37632, 48, 1), 0), out=buf1175)
        buf1176 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf1177 = empty_strided((8, 16, 48, 1), (768, 48, 1, 6144), device='cuda', dtype=torch.float32)
        buf1178 = empty((8, 16, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_69, attn_70, attn_71], Original ATen: [aten._softmax, aten.clone, aten.mul]
        triton_per_fused__softmax_clone_mul_26.run(buf1175, primals_94, buf1176, buf1177, buf1178, 6144, 48, grid=grid(6144), stream=stream0)
        buf1179 = empty((8, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_47], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf1166, buf1179, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1180 = empty((128, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_47], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1178, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf1179, (128, 48, 784), (37632, 784, 1), 0), out=buf1180)
        buf1181 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_443], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_28.run(buf1180, buf1181, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf1182 = reinterpret_tensor(buf1180, (6272, 768), (768, 1), 0); del buf1180  # reuse
        # Source Nodes: [x_443], Original ATen: [aten.mm]
        extern_kernels.mm(buf1181, reinterpret_tensor(primals_577, (768, 768), (1, 768), 0), out=buf1182)
        buf1183 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf1187 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf1188 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf1282 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_23_norm3, mul_94, mul_96, x_441, x_443, x_445, x_446], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_36.run(buf1152, primals_92, buf1160, primals_93, buf1182, primals_578, primals_579, primals_580, buf1183, buf1187, buf1188, buf1282, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_580
        # Source Nodes: [x_447], Original ATen: [aten.convolution]
        buf1189 = extern_kernels.convolution(buf1188, primals_581, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf1189, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf1190 = reinterpret_tensor(buf1152, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf1152  # reuse
        # Source Nodes: [x_447], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf1189, primals_582, buf1190, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_582
        buf1191 = buf1144; del buf1144  # reuse
        buf1192 = buf1143; del buf1143  # reuse
        buf1193 = buf1142; del buf1142  # reuse
        # Source Nodes: [x_448, x_449], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_red_fused__native_batch_norm_legit_functional_gelu_31.run(buf1190, buf1191, buf1192, buf1193, 37632, 128, grid=grid(37632), stream=stream0)
        buf1194 = buf1146; del buf1146  # reuse
        buf1195 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf1197 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_448, x_449], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf1191, buf1192, buf1193, primals_707, primals_708, buf1194, buf1195, buf1197, primals_707, primals_708, 768, 49, grid=grid(768), stream=stream0)
        del buf1191
        del buf1192
        del buf1193
        del primals_707
        del primals_708
        buf1198 = reinterpret_tensor(buf1189, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf1189  # reuse
        # Source Nodes: [x_448, x_449], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_functional_gelu_32.run(buf1190, buf1194, buf1195, primals_583, primals_584, buf1198, 4816896, grid=grid(4816896), stream=stream0)
        del buf1195
        del primals_584
        # Source Nodes: [x_450], Original ATen: [aten.convolution]
        buf1199 = extern_kernels.convolution(buf1198, primals_585, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf1199, (8, 768, 28, 28), (602112, 784, 28, 1))
        buf1200 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_450], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf1199, primals_586, buf1200, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del primals_586
        buf1204 = reinterpret_tensor(buf1199, (8, 784, 768), (602112, 768, 1), 0); del buf1199  # reuse
        buf1205 = empty((6272, 768), device='cuda', dtype=torch.float32)
        buf1281 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_23_norm2, mul_97, x_452, x_453], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_35.run(buf1183, primals_95, buf1200, primals_587, primals_588, buf1204, buf1205, buf1281, 6272, 768, grid=grid(6272), stream=stream0)
        del primals_588
        buf1206 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_453], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_590, buf1205, reinterpret_tensor(primals_589, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf1206)
        del primals_590
        buf1207 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_454, x_457], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_34.run(buf1206, buf1207, 19267584, grid=grid(19267584), stream=stream0)
        buf1208 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_457], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_592, buf1207, reinterpret_tensor(primals_591, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf1208)
        del primals_592
        buf1209 = empty((8, 785, 768), device='cuda', dtype=torch.float32)
        buf1210 = empty((8, 785, 1), device='cuda', dtype=torch.float32)
        buf1211 = empty_strided((8, 785, 1), (785, 1, 6280), device='cuda', dtype=torch.float32)
        buf1213 = reinterpret_tensor(buf1211, (8, 785, 1), (785, 1, 1), 0); del buf1211  # reuse
        buf1214 = empty((8, 785, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_10, x_norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_38.run(buf1213, primals_97, buf1183, primals_95, buf1200, primals_96, buf1208, primals_593, primals_594, buf1209, buf1210, buf1214, 6280, 768, grid=grid(6280), stream=stream0)
        del buf1183
        del primals_594
        del primals_97
        buf1215 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___cls_attn_blocks_0_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_596, reinterpret_tensor(buf1214, (8, 768), (602880, 1), 0), reinterpret_tensor(primals_595, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf1215)
        del primals_596
        buf1216 = empty_strided((8, 16, 1, 48), (768, 1, 768, 16), device='cuda', dtype=torch.float32)
        buf1221 = empty_strided((8, 16, 1, 48), (768, 48, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_48, x_cls], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_39.run(buf1215, buf1216, buf1221, 128, 48, grid=grid(128, 48), stream=stream0)
        buf1217 = empty((6280, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___cls_attn_blocks_0_attn_k], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_598, reinterpret_tensor(buf1214, (6280, 768), (768, 1), 0), reinterpret_tensor(primals_597, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf1217)
        del primals_598
        buf1218 = empty_strided((8, 16, 785, 48), (602880, 1, 768, 16), device='cuda', dtype=torch.float32)
        buf1222 = empty_strided((8, 16, 785, 48), (602880, 48, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_48, x_cls], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_40.run(buf1217, buf1218, buf1222, 100480, 48, grid=grid(100480, 48), stream=stream0)
        buf1219 = buf1217; del buf1217  # reuse
        # Source Nodes: [l__mod___cls_attn_blocks_0_attn_v], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_600, reinterpret_tensor(buf1214, (6280, 768), (768, 1), 0), reinterpret_tensor(primals_599, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf1219)
        del primals_600
        buf1220 = empty_strided((8, 16, 785, 48), (602880, 1, 768, 16), device='cuda', dtype=torch.float32)
        buf1223 = empty_strided((8, 16, 785, 48), (602880, 48, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_24, x_cls], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_40.run(buf1219, buf1220, buf1223, 100480, 48, grid=grid(100480, 48), stream=stream0)
        # Source Nodes: [x_cls], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf1224 = aten._scaled_dot_product_efficient_attention(buf1221, buf1222, buf1223, None, True)
        buf1225 = buf1224[0]
        buf1226 = buf1224[1]
        buf1227 = buf1224[2]
        buf1228 = buf1224[3]
        del buf1224
        buf1229 = reinterpret_tensor(buf1221, (8, 768), (768, 1), 0); del buf1221  # reuse
        # Source Nodes: [x_cls_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_602, reinterpret_tensor(buf1225, (8, 768), (768, 1), 0), reinterpret_tensor(primals_601, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf1229)
        del primals_602
        buf1230 = reinterpret_tensor(buf1223, (8, 785, 768), (602880, 768, 1), 0); del buf1223  # reuse
        buf1234 = reinterpret_tensor(buf1222, (8, 785, 768), (602880, 768, 1), 0); del buf1222  # reuse
        buf1235 = reinterpret_tensor(buf1219, (8, 785, 768), (602880, 768, 1), 0); del buf1219  # reuse
        buf1279 = empty((8, 785, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_9, mul_99, x_462, x_res], Original ATen: [aten.add, aten.cat, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_cat_mul_native_layer_norm_native_layer_norm_backward_41.run(buf1229, buf1214, buf1209, primals_98, primals_603, primals_604, buf1230, buf1234, buf1235, buf1279, 6280, 768, grid=grid(6280), stream=stream0)
        del primals_604
        buf1236 = empty((8, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_464], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1235, (8, 768), (602880, 1), 0), reinterpret_tensor(primals_605, (768, 3072), (1, 768), 0), out=buf1236)
        buf1237 = empty((8, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_464, x_465, x_468], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_42.run(buf1236, primals_606, buf1237, 24576, grid=grid(24576), stream=stream0)
        buf1238 = buf1229; del buf1229  # reuse
        # Source Nodes: [x_468], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_608, buf1237, reinterpret_tensor(primals_607, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf1238)
        del primals_608
        buf1242 = empty((8, 785, 768), device='cuda', dtype=torch.float32)
        buf1243 = empty((8, 785, 768), device='cuda', dtype=torch.float32)
        buf1278 = empty((8, 785, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_8, x_472, x_norm1_1], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_43.run(buf1235, primals_99, buf1238, primals_609, primals_610, buf1242, buf1243, buf1278, 6280, 768, grid=grid(6280), stream=stream0)
        del primals_610
        buf1244 = buf1215; del buf1215  # reuse
        # Source Nodes: [l__mod___cls_attn_blocks_1_attn_q], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_612, reinterpret_tensor(buf1243, (8, 768), (602880, 1), 0), reinterpret_tensor(primals_611, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf1244)
        del primals_612
        buf1245 = empty_strided((8, 16, 1, 48), (768, 1, 768, 16), device='cuda', dtype=torch.float32)
        buf1250 = empty_strided((8, 16, 1, 48), (768, 48, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_49, x_cls_4], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_39.run(buf1244, buf1245, buf1250, 128, 48, grid=grid(128, 48), stream=stream0)
        buf1246 = empty((6280, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___cls_attn_blocks_1_attn_k], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_614, reinterpret_tensor(buf1243, (6280, 768), (768, 1), 0), reinterpret_tensor(primals_613, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf1246)
        del primals_614
        buf1247 = empty_strided((8, 16, 785, 48), (602880, 1, 768, 16), device='cuda', dtype=torch.float32)
        buf1251 = empty_strided((8, 16, 785, 48), (602880, 48, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_49, x_cls_4], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_40.run(buf1246, buf1247, buf1251, 100480, 48, grid=grid(100480, 48), stream=stream0)
        buf1248 = buf1246; del buf1246  # reuse
        # Source Nodes: [l__mod___cls_attn_blocks_1_attn_v], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_616, reinterpret_tensor(buf1243, (6280, 768), (768, 1), 0), reinterpret_tensor(primals_615, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf1248)
        del primals_616
        buf1249 = empty_strided((8, 16, 785, 48), (602880, 1, 768, 16), device='cuda', dtype=torch.float32)
        buf1252 = empty_strided((8, 16, 785, 48), (602880, 48, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_25, x_cls_4], Original ATen: [aten._scaled_dot_product_efficient_attention, aten.permute]
        triton_poi_fused__scaled_dot_product_efficient_attention_permute_40.run(buf1248, buf1249, buf1252, 100480, 48, grid=grid(100480, 48), stream=stream0)
        # Source Nodes: [x_cls_4], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf1253 = aten._scaled_dot_product_efficient_attention(buf1250, buf1251, buf1252, None, True)
        buf1254 = buf1253[0]
        buf1255 = buf1253[1]
        buf1256 = buf1253[2]
        buf1257 = buf1253[3]
        del buf1253
        buf1258 = reinterpret_tensor(buf1250, (8, 768), (768, 1), 0); del buf1250  # reuse
        # Source Nodes: [x_cls_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_618, reinterpret_tensor(buf1254, (8, 768), (768, 1), 0), reinterpret_tensor(primals_617, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf1258)
        del primals_618
        buf1259 = reinterpret_tensor(buf1252, (8, 785, 768), (602880, 768, 1), 0); del buf1252  # reuse
        buf1260 = reinterpret_tensor(buf1251, (8, 785, 768), (602880, 768, 1), 0); del buf1251  # reuse
        buf1264 = reinterpret_tensor(buf1248, (8, 785, 768), (602880, 768, 1), 0); del buf1248  # reuse
        buf1265 = empty((8, 785, 768), device='cuda', dtype=torch.float32)
        buf1276 = empty((8, 785, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_7, cat_8, mul_101, x_472, x_473, x_res_1], Original ATen: [aten.add, aten.cat, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_cat_mul_native_layer_norm_native_layer_norm_backward_44.run(buf1258, buf1243, buf1235, primals_99, buf1238, primals_100, primals_619, primals_620, buf1259, buf1260, buf1264, buf1265, buf1276, 6280, 768, grid=grid(6280), stream=stream0)
        del primals_620
        buf1266 = empty((8, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_475], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1265, (8, 768), (602880, 1), 0), reinterpret_tensor(primals_621, (768, 3072), (1, 768), 0), out=buf1266)
        buf1267 = empty((8, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_475, x_476, x_479], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_42.run(buf1266, primals_622, buf1267, 24576, grid=grid(24576), stream=stream0)
        buf1268 = buf1258; del buf1258  # reuse
        # Source Nodes: [x_479], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_624, buf1267, reinterpret_tensor(primals_623, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf1268)
        del primals_624
        buf1272 = buf1260; del buf1260  # reuse
        buf1275 = empty((8, 785, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_6, x_483, x_485], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_45.run(buf1265, primals_101, buf1268, buf1272, buf1275, 6280, 768, grid=grid(6280), stream=stream0)
        buf1273 = buf1244; del buf1244  # reuse
        # Source Nodes: [x_487], Original ATen: [aten.clone]
        triton_poi_fused_clone_46.run(buf1272, primals_625, primals_626, buf1273, 6144, grid=grid(6144), stream=stream0)
        del primals_626
        buf1274 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_628, buf1273, reinterpret_tensor(primals_627, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf1274)
        del primals_628
        buf1277 = empty_strided((8, 16, 1, 48), (768, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_47.run(buf1254, buf1277, 128, 48, grid=grid(128, 48), stream=stream0)
        buf1280 = empty_strided((8, 16, 1, 48), (768, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.detach]
        triton_poi_fused_detach_47.run(buf1225, buf1280, 128, 48, grid=grid(128, 48), stream=stream0)
        buf1283 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_69, attn_70], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf1175, primals_94, buf1176, buf1177, buf1283, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf1176
        del buf1177
        buf1287 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_66, attn_67], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf1127, primals_90, buf1128, buf1129, buf1287, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf1128
        del buf1129
        buf1291 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_63, attn_64], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf1078, primals_86, buf1079, buf1080, buf1291, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf1079
        del buf1080
        buf1295 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_60, attn_61], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf1030, primals_82, buf1031, buf1032, buf1295, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf1031
        del buf1032
        buf1299 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_57, attn_58], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf981, primals_78, buf982, buf983, buf1299, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf982
        del buf983
        buf1303 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_54, attn_55], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf933, primals_74, buf934, buf935, buf1303, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf934
        del buf935
        buf1307 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_51, attn_52], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf884, primals_70, buf885, buf886, buf1307, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf885
        del buf886
        buf1311 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_48, attn_49], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf836, primals_66, buf837, buf838, buf1311, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf837
        del buf838
        buf1315 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_45, attn_46], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf787, primals_62, buf788, buf789, buf1315, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf788
        del buf789
        buf1319 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_42, attn_43], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf739, primals_58, buf740, buf741, buf1319, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf740
        del buf741
        buf1323 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_39, attn_40], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf690, primals_54, buf691, buf692, buf1323, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf691
        del buf692
        buf1327 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_36, attn_37], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf642, primals_50, buf643, buf644, buf1327, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf643
        del buf644
        buf1331 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_33, attn_34], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf593, primals_46, buf594, buf595, buf1331, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf594
        del buf595
        buf1335 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_30, attn_31], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf545, primals_42, buf546, buf547, buf1335, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf546
        del buf547
        buf1339 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_27, attn_28], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf496, primals_38, buf497, buf498, buf1339, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf497
        del buf498
        buf1343 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_24, attn_25], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf448, primals_34, buf449, buf450, buf1343, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf449
        del buf450
        buf1347 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_21, attn_22], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf399, primals_30, buf400, buf401, buf1347, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf400
        del buf401
        buf1351 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_18, attn_19], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf351, primals_26, buf352, buf353, buf1351, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf352
        del buf353
        buf1355 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_15, attn_16], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf302, primals_22, buf303, buf304, buf1355, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf303
        del buf304
        buf1359 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_12, attn_13], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf254, primals_18, buf255, buf256, buf1359, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf255
        del buf256
        buf1363 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_10, attn_9], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf205, primals_14, buf206, buf207, buf1363, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf206
        del buf207
        buf1367 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_6, attn_7], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf157, primals_10, buf158, buf159, buf1367, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf158
        del buf159
        buf1371 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_3, attn_4], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf108, primals_6, buf109, buf110, buf1371, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf109
        del buf110
        buf1375 = empty_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn, attn_1], Original ATen: [aten._softmax, aten.detach, aten.mul]
        triton_poi_fused__softmax_detach_mul_48.run(buf60, primals_2, buf61, buf62, buf1375, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del buf61
        del buf62
        # Source Nodes: [l__mod___patch_embed_proj_0_1], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_631, primals_631, 1, grid=grid(1), stream=stream0)
        del primals_631
        # Source Nodes: [l__mod___patch_embed_proj_2_1], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_634, primals_634, 1, grid=grid(1), stream=stream0)
        del primals_634
        # Source Nodes: [x], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_637, primals_637, 1, grid=grid(1), stream=stream0)
        del primals_637
        # Source Nodes: [x_12], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_640, primals_640, 1, grid=grid(1), stream=stream0)
        del primals_640
        # Source Nodes: [x_31], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_643, primals_643, 1, grid=grid(1), stream=stream0)
        del primals_643
        # Source Nodes: [x_50], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_646, primals_646, 1, grid=grid(1), stream=stream0)
        del primals_646
        # Source Nodes: [x_69], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_649, primals_649, 1, grid=grid(1), stream=stream0)
        del primals_649
        # Source Nodes: [x_88], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_652, primals_652, 1, grid=grid(1), stream=stream0)
        del primals_652
        # Source Nodes: [x_107], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_655, primals_655, 1, grid=grid(1), stream=stream0)
        del primals_655
        # Source Nodes: [x_126], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_658, primals_658, 1, grid=grid(1), stream=stream0)
        del primals_658
        # Source Nodes: [x_145], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_661, primals_661, 1, grid=grid(1), stream=stream0)
        del primals_661
        # Source Nodes: [x_164], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_664, primals_664, 1, grid=grid(1), stream=stream0)
        del primals_664
        # Source Nodes: [x_183], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_667, primals_667, 1, grid=grid(1), stream=stream0)
        del primals_667
        # Source Nodes: [x_202], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_670, primals_670, 1, grid=grid(1), stream=stream0)
        del primals_670
        # Source Nodes: [x_221], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_673, primals_673, 1, grid=grid(1), stream=stream0)
        del primals_673
        # Source Nodes: [x_240], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_676, primals_676, 1, grid=grid(1), stream=stream0)
        del primals_676
        # Source Nodes: [x_259], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_679, primals_679, 1, grid=grid(1), stream=stream0)
        del primals_679
        # Source Nodes: [x_278], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_682, primals_682, 1, grid=grid(1), stream=stream0)
        del primals_682
        # Source Nodes: [x_297], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_685, primals_685, 1, grid=grid(1), stream=stream0)
        del primals_685
        # Source Nodes: [x_316], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_688, primals_688, 1, grid=grid(1), stream=stream0)
        del primals_688
        # Source Nodes: [x_335], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_691, primals_691, 1, grid=grid(1), stream=stream0)
        del primals_691
        # Source Nodes: [x_354], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_694, primals_694, 1, grid=grid(1), stream=stream0)
        del primals_694
        # Source Nodes: [x_373], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_697, primals_697, 1, grid=grid(1), stream=stream0)
        del primals_697
        # Source Nodes: [x_392], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_700, primals_700, 1, grid=grid(1), stream=stream0)
        del primals_700
        # Source Nodes: [x_411], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_703, primals_703, 1, grid=grid(1), stream=stream0)
        del primals_703
        # Source Nodes: [x_430], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_706, primals_706, 1, grid=grid(1), stream=stream0)
        del primals_706
        # Source Nodes: [x_449], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(primals_709, primals_709, 1, grid=grid(1), stream=stream0)
        del primals_709
        return (buf1274, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_98, primals_99, primals_100, primals_101, buf0, primals_103, buf1, primals_106, buf2, primals_109, primals_111, primals_113, primals_118, primals_119, primals_121, primals_123, primals_125, primals_127, primals_133, primals_138, primals_139, primals_141, primals_143, primals_145, primals_147, primals_153, primals_158, primals_159, primals_161, primals_163, primals_165, primals_167, primals_173, primals_178, primals_179, primals_181, primals_183, primals_185, primals_187, primals_193, primals_198, primals_199, primals_201, primals_203, primals_205, primals_207, primals_213, primals_218, primals_219, primals_221, primals_223, primals_225, primals_227, primals_233, primals_238, primals_239, primals_241, primals_243, primals_245, primals_247, primals_253, primals_258, primals_259, primals_261, primals_263, primals_265, primals_267, primals_273, primals_278, primals_279, primals_281, primals_283, primals_285, primals_287, primals_293, primals_298, primals_299, primals_301, primals_303, primals_305, primals_307, primals_313, primals_318, primals_319, primals_321, primals_323, primals_325, primals_327, primals_333, primals_338, primals_339, primals_341, primals_343, primals_345, primals_347, primals_353, primals_358, primals_359, primals_361, primals_363, primals_365, primals_367, primals_373, primals_378, primals_379, primals_381, primals_383, primals_385, primals_387, primals_393, primals_398, primals_399, primals_401, primals_403, primals_405, primals_407, primals_413, primals_418, primals_419, primals_421, primals_423, primals_425, primals_427, primals_433, primals_438, primals_439, primals_441, primals_443, primals_445, primals_447, primals_453, primals_458, primals_459, primals_461, primals_463, primals_465, primals_467, primals_473, primals_478, primals_479, primals_481, primals_483, primals_485, primals_487, primals_493, primals_498, primals_499, primals_501, primals_503, primals_505, primals_507, primals_513, primals_518, primals_519, primals_521, primals_523, primals_525, primals_527, primals_533, primals_538, primals_539, primals_541, primals_543, primals_545, primals_547, primals_553, primals_558, primals_559, primals_561, primals_563, primals_565, primals_567, primals_573, primals_578, primals_579, primals_581, primals_583, primals_585, primals_587, primals_593, primals_603, primals_606, primals_609, primals_619, primals_622, primals_625, buf3, buf5, buf15, buf17, buf19, buf29, buf31, buf33, buf40, buf43, buf49, buf50, reinterpret_tensor(buf51, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf51, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf54, buf57, buf60, buf66, buf67, buf71, buf72, buf74, buf81, buf82, buf84, buf89, buf90, buf91, buf92, buf93, buf97, buf98, reinterpret_tensor(buf99, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf99, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf102, buf105, buf108, buf114, buf115, buf120, buf121, buf123, buf130, buf131, buf133, buf137, buf138, buf139, buf140, buf141, buf146, buf147, reinterpret_tensor(buf148, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf148, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf151, buf154, buf157, buf163, buf164, buf168, buf169, buf171, buf178, buf179, buf181, buf186, buf187, buf188, buf189, buf190, buf194, buf195, reinterpret_tensor(buf196, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf196, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf199, buf202, buf205, buf211, buf212, buf217, buf218, buf220, buf227, buf228, buf230, buf234, buf235, buf236, buf237, buf238, buf243, buf244, reinterpret_tensor(buf245, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf245, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf248, buf251, buf254, buf260, buf261, buf265, buf266, buf268, buf275, buf276, buf278, buf283, buf284, buf285, buf286, buf287, buf291, buf292, reinterpret_tensor(buf293, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf293, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf296, buf299, buf302, buf308, buf309, buf314, buf315, buf317, buf324, buf325, buf327, buf331, buf332, buf333, buf334, buf335, buf340, buf341, reinterpret_tensor(buf342, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf342, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf345, buf348, buf351, buf357, buf358, buf362, buf363, buf365, buf372, buf373, buf375, buf380, buf381, buf382, buf383, buf384, buf388, buf389, reinterpret_tensor(buf390, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf390, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf393, buf396, buf399, buf405, buf406, buf411, buf412, buf414, buf421, buf422, buf424, buf428, buf429, buf430, buf431, buf432, buf437, buf438, reinterpret_tensor(buf439, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf439, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf442, buf445, buf448, buf454, buf455, buf459, buf460, buf462, buf469, buf470, buf472, buf477, buf478, buf479, buf480, buf481, buf485, buf486, reinterpret_tensor(buf487, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf487, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf490, buf493, buf496, buf502, buf503, buf508, buf509, buf511, buf518, buf519, buf521, buf525, buf526, buf527, buf528, buf529, buf534, buf535, reinterpret_tensor(buf536, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf536, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf539, buf542, buf545, buf551, buf552, buf556, buf557, buf559, buf566, buf567, buf569, buf574, buf575, buf576, buf577, buf578, buf582, buf583, reinterpret_tensor(buf584, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf584, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf587, buf590, buf593, buf599, buf600, buf605, buf606, buf608, buf615, buf616, buf618, buf622, buf623, buf624, buf625, buf626, buf631, buf632, reinterpret_tensor(buf633, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf633, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf636, buf639, buf642, buf648, buf649, buf653, buf654, buf656, buf663, buf664, buf666, buf671, buf672, buf673, buf674, buf675, buf679, buf680, reinterpret_tensor(buf681, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf681, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf684, buf687, buf690, buf696, buf697, buf702, buf703, buf705, buf712, buf713, buf715, buf719, buf720, buf721, buf722, buf723, buf728, buf729, reinterpret_tensor(buf730, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf730, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf733, buf736, buf739, buf745, buf746, buf750, buf751, buf753, buf760, buf761, buf763, buf768, buf769, buf770, buf771, buf772, buf776, buf777, reinterpret_tensor(buf778, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf778, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf781, buf784, buf787, buf793, buf794, buf799, buf800, buf802, buf809, buf810, buf812, buf816, buf817, buf818, buf819, buf820, buf825, buf826, reinterpret_tensor(buf827, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf827, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf830, buf833, buf836, buf842, buf843, buf847, buf848, buf850, buf857, buf858, buf860, buf865, buf866, buf867, buf868, buf869, buf873, buf874, reinterpret_tensor(buf875, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf875, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf878, buf881, buf884, buf890, buf891, buf896, buf897, buf899, buf906, buf907, buf909, buf913, buf914, buf915, buf916, buf917, buf922, buf923, reinterpret_tensor(buf924, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf924, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf927, buf930, buf933, buf939, buf940, buf944, buf945, buf947, buf954, buf955, buf957, buf962, buf963, buf964, buf965, buf966, buf970, buf971, reinterpret_tensor(buf972, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf972, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf975, buf978, buf981, buf987, buf988, buf993, buf994, buf996, buf1003, buf1004, buf1006, buf1010, buf1011, buf1012, buf1013, buf1014, buf1019, buf1020, reinterpret_tensor(buf1021, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf1021, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf1024, buf1027, buf1030, buf1036, buf1037, buf1041, buf1042, buf1044, buf1051, buf1052, buf1054, buf1059, buf1060, buf1061, buf1062, buf1063, buf1067, buf1068, reinterpret_tensor(buf1069, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf1069, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf1072, buf1075, buf1078, buf1084, buf1085, buf1090, buf1091, buf1093, buf1100, buf1101, buf1103, buf1107, buf1108, buf1109, buf1110, buf1111, buf1116, buf1117, reinterpret_tensor(buf1118, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf1118, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf1121, buf1124, buf1127, buf1133, buf1134, buf1138, buf1139, buf1141, buf1148, buf1149, buf1151, buf1156, buf1157, buf1158, buf1159, buf1160, buf1164, buf1165, reinterpret_tensor(buf1166, (8, 16, 48, 784), (1806336, 48, 1, 2304), 0), reinterpret_tensor(buf1166, (8, 16, 48, 784), (1806336, 48, 1, 2304), 768), buf1169, buf1172, buf1175, buf1181, buf1182, buf1187, buf1188, buf1190, buf1197, buf1198, buf1200, buf1204, buf1205, buf1206, buf1207, buf1208, buf1209, buf1210, buf1213, reinterpret_tensor(buf1214, (8, 768), (602880, 1), 0), buf1216, reinterpret_tensor(buf1214, (6280, 768), (768, 1), 0), buf1218, buf1220, buf1226, buf1227, buf1228, reinterpret_tensor(buf1225, (8, 768), (768, 1), 0), buf1230, buf1234, reinterpret_tensor(buf1235, (8, 768), (602880, 1), 0), buf1236, buf1237, buf1238, buf1242, reinterpret_tensor(buf1243, (8, 768), (602880, 1), 0), buf1245, reinterpret_tensor(buf1243, (6280, 768), (768, 1), 0), buf1247, buf1249, buf1255, buf1256, buf1257, reinterpret_tensor(buf1254, (8, 768), (768, 1), 0), buf1259, buf1264, reinterpret_tensor(buf1265, (8, 768), (602880, 1), 0), buf1266, buf1267, buf1268, buf1272, buf1273, reinterpret_tensor(primals_627, (1000, 768), (768, 1), 0), buf1275, reinterpret_tensor(primals_623, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_621, (3072, 768), (768, 1), 0), buf1276, reinterpret_tensor(primals_617, (768, 768), (768, 1), 0), buf1277, reinterpret_tensor(primals_615, (768, 768), (768, 1), 0), reinterpret_tensor(primals_613, (768, 768), (768, 1), 0), reinterpret_tensor(primals_611, (768, 768), (768, 1), 0), buf1278, reinterpret_tensor(primals_607, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_605, (3072, 768), (768, 1), 0), buf1279, reinterpret_tensor(primals_601, (768, 768), (768, 1), 0), buf1280, reinterpret_tensor(primals_599, (768, 768), (768, 1), 0), reinterpret_tensor(primals_597, (768, 768), (768, 1), 0), reinterpret_tensor(primals_595, (768, 768), (768, 1), 0), reinterpret_tensor(primals_591, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_589, (3072, 768), (768, 1), 0), buf1281, reinterpret_tensor(buf1194, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1282, reinterpret_tensor(primals_577, (768, 768), (768, 1), 0), reinterpret_tensor(buf1178, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf1179, (128, 784, 48), (37632, 1, 784), 0), buf1283, reinterpret_tensor(buf1173, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf1174, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_575, (2304, 768), (768, 1), 0), buf1284, reinterpret_tensor(primals_571, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_569, (3072, 768), (768, 1), 0), buf1285, reinterpret_tensor(buf1145, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1286, reinterpret_tensor(primals_557, (768, 768), (768, 1), 0), reinterpret_tensor(buf1130, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf1131, (128, 784, 48), (37632, 1, 784), 0), buf1287, reinterpret_tensor(buf1125, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf1126, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_555, (2304, 768), (768, 1), 0), buf1288, reinterpret_tensor(primals_551, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_549, (3072, 768), (768, 1), 0), buf1289, reinterpret_tensor(buf1097, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1290, reinterpret_tensor(primals_537, (768, 768), (768, 1), 0), reinterpret_tensor(buf1081, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf1082, (128, 784, 48), (37632, 1, 784), 0), buf1291, reinterpret_tensor(buf1076, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf1077, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_535, (2304, 768), (768, 1), 0), buf1292, reinterpret_tensor(primals_531, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_529, (3072, 768), (768, 1), 0), buf1293, reinterpret_tensor(buf1048, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1294, reinterpret_tensor(primals_517, (768, 768), (768, 1), 0), reinterpret_tensor(buf1033, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf1034, (128, 784, 48), (37632, 1, 784), 0), buf1295, reinterpret_tensor(buf1028, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf1029, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_515, (2304, 768), (768, 1), 0), buf1296, reinterpret_tensor(primals_511, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_509, (3072, 768), (768, 1), 0), buf1297, reinterpret_tensor(buf1000, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1298, reinterpret_tensor(primals_497, (768, 768), (768, 1), 0), reinterpret_tensor(buf984, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf985, (128, 784, 48), (37632, 1, 784), 0), buf1299, reinterpret_tensor(buf979, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf980, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_495, (2304, 768), (768, 1), 0), buf1300, reinterpret_tensor(primals_491, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_489, (3072, 768), (768, 1), 0), buf1301, reinterpret_tensor(buf951, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1302, reinterpret_tensor(primals_477, (768, 768), (768, 1), 0), reinterpret_tensor(buf936, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf937, (128, 784, 48), (37632, 1, 784), 0), buf1303, reinterpret_tensor(buf931, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf932, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_475, (2304, 768), (768, 1), 0), buf1304, reinterpret_tensor(primals_471, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_469, (3072, 768), (768, 1), 0), buf1305, reinterpret_tensor(buf903, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1306, reinterpret_tensor(primals_457, (768, 768), (768, 1), 0), reinterpret_tensor(buf887, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf888, (128, 784, 48), (37632, 1, 784), 0), buf1307, reinterpret_tensor(buf882, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf883, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_455, (2304, 768), (768, 1), 0), buf1308, reinterpret_tensor(primals_451, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_449, (3072, 768), (768, 1), 0), buf1309, reinterpret_tensor(buf854, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1310, reinterpret_tensor(primals_437, (768, 768), (768, 1), 0), reinterpret_tensor(buf839, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf840, (128, 784, 48), (37632, 1, 784), 0), buf1311, reinterpret_tensor(buf834, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf835, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_435, (2304, 768), (768, 1), 0), buf1312, reinterpret_tensor(primals_431, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_429, (3072, 768), (768, 1), 0), buf1313, reinterpret_tensor(buf806, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1314, reinterpret_tensor(primals_417, (768, 768), (768, 1), 0), reinterpret_tensor(buf790, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf791, (128, 784, 48), (37632, 1, 784), 0), buf1315, reinterpret_tensor(buf785, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf786, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_415, (2304, 768), (768, 1), 0), buf1316, reinterpret_tensor(primals_411, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_409, (3072, 768), (768, 1), 0), buf1317, reinterpret_tensor(buf757, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1318, reinterpret_tensor(primals_397, (768, 768), (768, 1), 0), reinterpret_tensor(buf742, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf743, (128, 784, 48), (37632, 1, 784), 0), buf1319, reinterpret_tensor(buf737, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf738, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_395, (2304, 768), (768, 1), 0), buf1320, reinterpret_tensor(primals_391, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_389, (3072, 768), (768, 1), 0), buf1321, reinterpret_tensor(buf709, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1322, reinterpret_tensor(primals_377, (768, 768), (768, 1), 0), reinterpret_tensor(buf693, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf694, (128, 784, 48), (37632, 1, 784), 0), buf1323, reinterpret_tensor(buf688, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf689, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_375, (2304, 768), (768, 1), 0), buf1324, reinterpret_tensor(primals_371, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_369, (3072, 768), (768, 1), 0), buf1325, reinterpret_tensor(buf660, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1326, reinterpret_tensor(primals_357, (768, 768), (768, 1), 0), reinterpret_tensor(buf645, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf646, (128, 784, 48), (37632, 1, 784), 0), buf1327, reinterpret_tensor(buf640, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf641, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_355, (2304, 768), (768, 1), 0), buf1328, reinterpret_tensor(primals_351, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_349, (3072, 768), (768, 1), 0), buf1329, reinterpret_tensor(buf612, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1330, reinterpret_tensor(primals_337, (768, 768), (768, 1), 0), reinterpret_tensor(buf596, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf597, (128, 784, 48), (37632, 1, 784), 0), buf1331, reinterpret_tensor(buf591, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf592, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_335, (2304, 768), (768, 1), 0), buf1332, reinterpret_tensor(primals_331, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_329, (3072, 768), (768, 1), 0), buf1333, reinterpret_tensor(buf563, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1334, reinterpret_tensor(primals_317, (768, 768), (768, 1), 0), reinterpret_tensor(buf548, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf549, (128, 784, 48), (37632, 1, 784), 0), buf1335, reinterpret_tensor(buf543, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf544, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_315, (2304, 768), (768, 1), 0), buf1336, reinterpret_tensor(primals_311, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_309, (3072, 768), (768, 1), 0), buf1337, reinterpret_tensor(buf515, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1338, reinterpret_tensor(primals_297, (768, 768), (768, 1), 0), reinterpret_tensor(buf499, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf500, (128, 784, 48), (37632, 1, 784), 0), buf1339, reinterpret_tensor(buf494, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf495, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_295, (2304, 768), (768, 1), 0), buf1340, reinterpret_tensor(primals_291, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_289, (3072, 768), (768, 1), 0), buf1341, reinterpret_tensor(buf466, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1342, reinterpret_tensor(primals_277, (768, 768), (768, 1), 0), reinterpret_tensor(buf451, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf452, (128, 784, 48), (37632, 1, 784), 0), buf1343, reinterpret_tensor(buf446, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf447, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_275, (2304, 768), (768, 1), 0), buf1344, reinterpret_tensor(primals_271, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_269, (3072, 768), (768, 1), 0), buf1345, reinterpret_tensor(buf418, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1346, reinterpret_tensor(primals_257, (768, 768), (768, 1), 0), reinterpret_tensor(buf402, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf403, (128, 784, 48), (37632, 1, 784), 0), buf1347, reinterpret_tensor(buf397, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf398, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_255, (2304, 768), (768, 1), 0), buf1348, reinterpret_tensor(primals_251, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_249, (3072, 768), (768, 1), 0), buf1349, reinterpret_tensor(buf369, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1350, reinterpret_tensor(primals_237, (768, 768), (768, 1), 0), reinterpret_tensor(buf354, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf355, (128, 784, 48), (37632, 1, 784), 0), buf1351, reinterpret_tensor(buf349, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf350, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_235, (2304, 768), (768, 1), 0), buf1352, reinterpret_tensor(primals_231, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_229, (3072, 768), (768, 1), 0), buf1353, reinterpret_tensor(buf321, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1354, reinterpret_tensor(primals_217, (768, 768), (768, 1), 0), reinterpret_tensor(buf305, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf306, (128, 784, 48), (37632, 1, 784), 0), buf1355, reinterpret_tensor(buf300, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf301, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_215, (2304, 768), (768, 1), 0), buf1356, reinterpret_tensor(primals_211, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_209, (3072, 768), (768, 1), 0), buf1357, reinterpret_tensor(buf272, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1358, reinterpret_tensor(primals_197, (768, 768), (768, 1), 0), reinterpret_tensor(buf257, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf258, (128, 784, 48), (37632, 1, 784), 0), buf1359, reinterpret_tensor(buf252, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf253, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_195, (2304, 768), (768, 1), 0), buf1360, reinterpret_tensor(primals_191, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_189, (3072, 768), (768, 1), 0), buf1361, reinterpret_tensor(buf224, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1362, reinterpret_tensor(primals_177, (768, 768), (768, 1), 0), reinterpret_tensor(buf208, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf209, (128, 784, 48), (37632, 1, 784), 0), buf1363, reinterpret_tensor(buf203, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf204, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_175, (2304, 768), (768, 1), 0), buf1364, reinterpret_tensor(primals_171, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_169, (3072, 768), (768, 1), 0), buf1365, reinterpret_tensor(buf175, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1366, reinterpret_tensor(primals_157, (768, 768), (768, 1), 0), reinterpret_tensor(buf160, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf161, (128, 784, 48), (37632, 1, 784), 0), buf1367, reinterpret_tensor(buf155, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf156, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_155, (2304, 768), (768, 1), 0), buf1368, reinterpret_tensor(primals_151, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_149, (3072, 768), (768, 1), 0), buf1369, reinterpret_tensor(buf127, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1370, reinterpret_tensor(primals_137, (768, 768), (768, 1), 0), reinterpret_tensor(buf111, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf112, (128, 784, 48), (37632, 1, 784), 0), buf1371, reinterpret_tensor(buf106, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf107, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_135, (2304, 768), (768, 1), 0), buf1372, reinterpret_tensor(primals_131, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_129, (3072, 768), (768, 1), 0), buf1373, reinterpret_tensor(buf78, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1374, reinterpret_tensor(primals_117, (768, 768), (768, 1), 0), reinterpret_tensor(buf63, (128, 48, 48), (2304, 1, 48), 0), reinterpret_tensor(buf64, (128, 784, 48), (37632, 1, 784), 0), buf1375, reinterpret_tensor(buf58, (128, 784, 48), (37632, 1, 784), 0), reinterpret_tensor(buf59, (128, 48, 784), (37632, 1, 48), 0), reinterpret_tensor(primals_115, (2304, 768), (768, 1), 0), buf1376, reinterpret_tensor(buf37, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf1377, reinterpret_tensor(buf26, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf1378, reinterpret_tensor(buf12, (1, 192, 1, 1), (192, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((192, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_632 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_635 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_638 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_641 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_644 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_647 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_650 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_653 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_656 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_659 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_662 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_665 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_668 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_671 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_674 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_677 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_680 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_683 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_684 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_686 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_687 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_689 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_690 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_692 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_693 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_695 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_696 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_698 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_699 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_701 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_702 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_704 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_705 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_707 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_708 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_710 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('xcit_large_24_p8_224', benchmark_compiled_module)
