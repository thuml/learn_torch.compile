
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


# kernel path: /tmp/torchinductor_youkaichao/o6/co66g27odg25ejip2t64kk2etmxjo3bilszuy5c4wq35kcoqu3an.py
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
    size_hints=[256, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (147*y1)), tmp0, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/xc/cxc67u7i6xam63uiya6j7n3eawqbby3stpthoca2l6aubd36psct.py
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
    size_hints=[1024, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 676
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 26
    y1 = (yindex // 26)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (26*x2) + (234*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ws/cwsoshyo34ywbrzdmxx255mlv3wvnynee4c4jd2ydhkmwlu2ppsg.py
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
    size_hints=[4096, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2704
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 52
    y1 = (yindex // 52)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (52*x2) + (468*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pe/cpencnugwhvwlypmqcectvefvijocpotbbiuz6oxvt5fftg664e5.py
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
    size_hints=[16384, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10816
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (104*x2) + (936*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ya/cya4e44ono7pvwxsxq56bsybbmkbgqn2e3w6e3wvqcw22c66ofai.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 43264
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (208*x2) + (1872*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qe/cqerugx5ktpzci54maukwqqsy73rg6imjytkyytnnaeorsbof2qz.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/iq/ciqgmk7ozxgw56x4vfhc7pj4zr7nx5f4ihbuu3ugcwikq4p5v47i.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 12544
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
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (802816*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ex/cexwaqjcvb5sdui4loevleqbkal3mmk22umoodb3b6ldip2jtrbn.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ad/cad22zoo5zg5mbpxzjr7vk2mtqj7ond4xz7jg63p3ldii6gvkcy6.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (7168*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (64*r2) + (7168*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (64*r2) + (7168*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (64*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (64*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (64*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hg/chgp3yxmhhnnehusxj44qqkibb5omh2uz7cql355to3ycjekmkkh.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => add_1, add_2, add_3, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze_1, var_mean
triton_per_fused__native_batch_norm_legit_functional_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_9', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (64*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/yf/cyfdkljjq3eco2csy3lydcfzgavndnjp44e5ko24oajzeurvixxk.py
# Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
# x_2 => relu
triton_poi_fused__native_batch_norm_legit_functional_relu_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uu/cuugatniwwg224f3y3lhjeoy5gv7q7ye6jnbsoc372po46kywate.py
# Source Nodes: [shortcut], Original ATen: [aten.max_pool2d_with_indices]
# shortcut => getitem_2, getitem_3
triton_poi_fused_max_pool2d_with_indices_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3584) % 56
    x1 = (xindex // 64) % 56
    x0 = xindex % 64
    x5 = (xindex // 3584)
    x6 = xindex
    tmp0 = (-1) + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x1)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-7232) + x0 + (128*x1) + (14336*x5)), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-7168) + x0 + (128*x1) + (14336*x5)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x1)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-7104) + x0 + (128*x1) + (14336*x5)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-64) + x0 + (128*x1) + (14336*x5)), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x0 + (128*x1) + (14336*x5)), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (14336*x5)), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 1 + (2*x2)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (7104 + x0 + (128*x1) + (14336*x5)), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (7168 + x0 + (128*x1) + (14336*x5)), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (7232 + x0 + (128*x1) + (14336*x5)), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tmp70 = tmp21 > tmp13
    tmp71 = (-112) + (2*x1) + (224*x2)
    tmp72 = (-113) + (2*x1) + (224*x2)
    tmp73 = tl.where(tmp70, tmp71, tmp72)
    tmp74 = tmp30 > tmp22
    tmp75 = (-111) + (2*x1) + (224*x2)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp77 = tmp39 > tmp31
    tmp78 = (-1) + (2*x1) + (224*x2)
    tmp79 = tl.where(tmp77, tmp78, tmp76)
    tmp80 = tmp44 > tmp40
    tmp81 = (2*x1) + (224*x2)
    tmp82 = tl.where(tmp80, tmp81, tmp79)
    tmp83 = tmp49 > tmp45
    tmp84 = 1 + (2*x1) + (224*x2)
    tmp85 = tl.where(tmp83, tmp84, tmp82)
    tmp86 = tmp58 > tmp50
    tmp87 = 111 + (2*x1) + (224*x2)
    tmp88 = tl.where(tmp86, tmp87, tmp85)
    tmp89 = tmp63 > tmp59
    tmp90 = 112 + (2*x1) + (224*x2)
    tmp91 = tl.where(tmp89, tmp90, tmp88)
    tmp92 = tmp68 > tmp64
    tmp93 = 113 + (2*x1) + (224*x2)
    tmp94 = tl.where(tmp92, tmp93, tmp91)
    tl.store(out_ptr0 + (x6), tmp69, None)
    tl.store(out_ptr1 + (x6), tmp94, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/iw/ciwoau7n6eedjzr2ymzu5bxwaoywwawzblymmokp6x6fvfeluygp.py
# Source Nodes: [out], Original ATen: [aten.convolution]
# out => convolution_1
triton_poi_fused_convolution_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (104*x2) + (326144*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/37/c375346bndqrmwtdjab4cvqi6gmj2e6nupgacqvcjsq5o6m7kxz2.py
# Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
# out_1 => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 20384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 104
    x1 = (xindex // 104)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (104*r2) + (13312*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/dd/cddbrysypodoqzst42oro26plffpfmaca3fsj4i7hiswu6emwpzu.py
# Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
# out_1 => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 208
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
        tmp0 = tl.load(in_ptr0 + (x1 + (104*r2) + (10192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (104*r2) + (10192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (104*r2) + (10192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (104*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (104*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (104*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tt/cttona3vwpyhp6s2amxftvn76qn7tpsueliialwky3ljbzbffamz.py
# Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
# out_1 => add_6, add_7, add_8, mul_10, mul_11, mul_12, mul_8, mul_9, rsqrt_1, squeeze_4, var_mean_1
triton_per_fused__native_batch_norm_legit_functional_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_15', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 104
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (104*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (104*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (104*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/k2/ck2ilvlhrdu6tbcfakz2qvc6nkxucvyedaszmgs5aegpxowfiptd.py
# Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_1 => add_6, add_9, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
# out_2 => relu_1
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2609152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 104
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ty/ctyihbzmvqgzhwdse3cr452hs5vkfwo42wge7q5q2ck4skzc3zuq.py
# Source Nodes: [sp_1], Original ATen: [aten.convolution]
# sp_1 => convolution_2
triton_poi_fused_convolution_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 208
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 26
    y1 = (yindex // 26)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (26*x2) + (81536*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dl/cdlcscqlfkumozrl2zo3u7bjhdtpg3slb7ytslbqjlm5sknvmhpi.py
# Source Nodes: [sp_2], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_2 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 26
    x1 = (xindex // 26)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (26*r2) + (3328*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/74/c74qgtv54cksukftooixrpecgqt5ej6js2civsp2uqncg7vpei62.py
# Source Nodes: [sp_2], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_2 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 52
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
        tmp0 = tl.load(in_ptr0 + (x1 + (26*r2) + (2548*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (26*r2) + (2548*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (26*r2) + (2548*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (26*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (26*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (26*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pl/cplgvpmmnmict7q4nescodvnoniqzu33e6nvfc44hbosr3s6ukv7.py
# Source Nodes: [sp_2], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_2 => add_11, add_12, add_13, mul_15, mul_16, mul_17, mul_18, mul_19, rsqrt_2, squeeze_7, var_mean_2
triton_per_fused__native_batch_norm_legit_functional_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 26
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (26*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (26*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (26*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/in/cingyi4qq2slf6zmz3jzhpdqm2nnxb5doby3ckvuyfv33far2im7.py
# Source Nodes: [sp_2, sp_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# sp_2 => add_11, add_14, mul_14, mul_20, rsqrt_2, sub_2, var_mean_2
# sp_3 => relu_2
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 208
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 26
    y1 = (yindex // 26)
    tmp0 = tl.load(in_ptr0 + (y0 + (26*x2) + (81536*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (3136*y0) + (326144*y1)), tmp14, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (26*x2) + (81536*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cp/ccpdfx3bperd6dydrgwywdq4jkmz76omjstikxod5mai74zegjdz.py
# Source Nodes: [getattr_l__mod___layer1___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer1___0___pool => avg_pool2d
triton_poi_fused_avg_pool2d_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 208
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 56)
    x2 = xindex % 56
    x5 = xindex
    y0 = yindex % 26
    y1 = (yindex // 26)
    tmp0 = (-1) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x2
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5850) + y0 + (104*x5) + (326144*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x2
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-5746) + y0 + (104*x5) + (326144*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x2
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-5642) + y0 + (104*x5) + (326144*y1)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x3
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-26) + y0 + (104*x5) + (326144*y1)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (78 + y0 + (104*x5) + (326144*y1)), tmp41 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (182 + y0 + (104*x5) + (326144*y1)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x3
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (5798 + y0 + (104*x5) + (326144*y1)), tmp55 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (5902 + y0 + (104*x5) + (326144*y1)), tmp60 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (6006 + y0 + (104*x5) + (326144*y1)), tmp65 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1, 1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1, 1], 57, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tl.broadcast_to((-1) + x3, [XBLOCK, YBLOCK])
    tmp80 = tmp79 >= tmp1
    tmp81 = tmp79 < tmp3
    tmp82 = tmp80 & tmp81
    tmp83 = tl.broadcast_to((-1) + x2, [XBLOCK, YBLOCK])
    tmp84 = tmp83 >= tmp1
    tmp85 = tmp83 < tmp3
    tmp86 = tmp84 & tmp85
    tmp87 = tmp82 & tmp86
    tmp88 = tmp87 & tmp78
    tmp89 = 1.0
    tmp90 = tl.full(tmp89.shape, 1.0, tmp89.dtype)
    tmp91 = tl.where(tmp88, tmp89, tmp90)
    tmp92 = tl.full(tmp91.shape, 0.0, tmp91.dtype)
    tmp93 = tl.where(tmp78, tmp91, tmp92)
    tmp94 = tmp14 >= tmp70
    tmp95 = tmp14 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tl.broadcast_to(x2, [XBLOCK, YBLOCK])
    tmp99 = tmp98 >= tmp1
    tmp100 = tmp98 < tmp3
    tmp101 = tmp99 & tmp100
    tmp102 = tmp82 & tmp101
    tmp103 = tmp102 & tmp97
    tmp104 = tl.where(tmp103, tmp89, tmp90)
    tmp105 = tl.full(tmp104.shape, 0.0, tmp104.dtype)
    tmp106 = tl.where(tmp97, tmp104, tmp105)
    tmp107 = tmp106 + tmp93
    tmp108 = tmp23 >= tmp70
    tmp109 = tmp23 < tmp72
    tmp110 = tmp108 & tmp109
    tmp111 = tmp74 & tmp110
    tmp112 = tl.broadcast_to(1 + x2, [XBLOCK, YBLOCK])
    tmp113 = tmp112 >= tmp1
    tmp114 = tmp112 < tmp3
    tmp115 = tmp113 & tmp114
    tmp116 = tmp82 & tmp115
    tmp117 = tmp116 & tmp111
    tmp118 = tl.where(tmp117, tmp89, tmp90)
    tmp119 = tl.full(tmp118.shape, 0.0, tmp118.dtype)
    tmp120 = tl.where(tmp111, tmp118, tmp119)
    tmp121 = tmp120 + tmp107
    tmp122 = tmp32 >= tmp70
    tmp123 = tmp32 < tmp72
    tmp124 = tmp122 & tmp123
    tmp125 = tmp124 & tmp77
    tmp126 = tl.broadcast_to(x3, [XBLOCK, YBLOCK])
    tmp127 = tmp126 >= tmp1
    tmp128 = tmp126 < tmp3
    tmp129 = tmp127 & tmp128
    tmp130 = tmp129 & tmp86
    tmp131 = tmp130 & tmp125
    tmp132 = tl.where(tmp131, tmp89, tmp90)
    tmp133 = tl.full(tmp132.shape, 0.0, tmp132.dtype)
    tmp134 = tl.where(tmp125, tmp132, tmp133)
    tmp135 = tmp134 + tmp121
    tmp136 = tmp124 & tmp96
    tmp137 = tmp129 & tmp101
    tmp138 = tmp137 & tmp136
    tmp139 = tl.where(tmp138, tmp89, tmp90)
    tmp140 = tl.full(tmp139.shape, 0.0, tmp139.dtype)
    tmp141 = tl.where(tmp136, tmp139, tmp140)
    tmp142 = tmp141 + tmp135
    tmp143 = tmp124 & tmp110
    tmp144 = tmp129 & tmp115
    tmp145 = tmp144 & tmp143
    tmp146 = tl.where(tmp145, tmp89, tmp90)
    tmp147 = tl.full(tmp146.shape, 0.0, tmp146.dtype)
    tmp148 = tl.where(tmp143, tmp146, tmp147)
    tmp149 = tmp148 + tmp142
    tmp150 = tmp51 >= tmp70
    tmp151 = tmp51 < tmp72
    tmp152 = tmp150 & tmp151
    tmp153 = tmp152 & tmp77
    tmp154 = tl.broadcast_to(1 + x3, [XBLOCK, YBLOCK])
    tmp155 = tmp154 >= tmp1
    tmp156 = tmp154 < tmp3
    tmp157 = tmp155 & tmp156
    tmp158 = tmp157 & tmp86
    tmp159 = tmp158 & tmp153
    tmp160 = tl.where(tmp159, tmp89, tmp90)
    tmp161 = tl.full(tmp160.shape, 0.0, tmp160.dtype)
    tmp162 = tl.where(tmp153, tmp160, tmp161)
    tmp163 = tmp162 + tmp149
    tmp164 = tmp152 & tmp96
    tmp165 = tmp157 & tmp101
    tmp166 = tmp165 & tmp164
    tmp167 = tl.where(tmp166, tmp89, tmp90)
    tmp168 = tl.full(tmp167.shape, 0.0, tmp167.dtype)
    tmp169 = tl.where(tmp164, tmp167, tmp168)
    tmp170 = tmp169 + tmp163
    tmp171 = tmp152 & tmp110
    tmp172 = tmp157 & tmp115
    tmp173 = tmp172 & tmp171
    tmp174 = tl.where(tmp173, tmp89, tmp90)
    tmp175 = tl.full(tmp174.shape, 0.0, tmp174.dtype)
    tmp176 = tl.where(tmp171, tmp174, tmp175)
    tmp177 = tmp176 + tmp170
    tmp178 = tmp69 / tmp177
    tl.store(out_ptr0 + (x5 + (3136*y0) + (326144*y1)), tmp178, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7j/c7jajy2iao6jhxbvsdcotaylopztqc27uby3xgqewdmciqzizx34.py
# Source Nodes: [out_4], Original ATen: [aten.convolution]
# out_4 => convolution_5
triton_poi_fused_convolution_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (802816*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5d/c5dqunrnocokumx6efxbuipwjjaji26s26ohwz22nu4qxvh46amo.py
# Source Nodes: [out_5], Original ATen: [aten._native_batch_norm_legit_functional]
# out_5 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/st/cst5jionhiymwhmf7nbd2xif6btwslcrziom2ftcxfvb57m2bto7.py
# Source Nodes: [out_5], Original ATen: [aten._native_batch_norm_legit_functional]
# out_5 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (25088*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (256*r2) + (25088*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (256*r2) + (25088*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (256*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (256*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (256*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q5/cq5dznw73ora3zzbtxabuxsgq2gmrvfjl5fnbx74qzkxqfsqqomu.py
# Source Nodes: [out_5], Original ATen: [aten._native_batch_norm_legit_functional]
# out_5 => add_26, add_27, add_28, mul_36, mul_37, mul_38, mul_39, mul_40, rsqrt_5, squeeze_16, var_mean_5
triton_per_fused__native_batch_norm_legit_functional_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_26', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (256*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ee/ceeadtlwb6567ut32bhb7nmje6yxtes5l66iuo7nnrzw2fcuu2hg.py
# Source Nodes: [out_5, out_6, shortcut_1, shortcut_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_5 => add_26, add_29, mul_35, mul_41, rsqrt_5, sub_5, var_mean_5
# out_6 => add_35
# shortcut_1 => add_31, add_34, mul_42, mul_48, rsqrt_6, sub_6, var_mean_6
# shortcut_2 => relu_5
triton_poi_fused__native_batch_norm_legit_functional_add_relu_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tb/ctbvyr4ep5omxzt4jad4z6uduicpuzd36ro5etr6dc5zleygjxgd.py
# Source Nodes: [sp_17], Original ATen: [aten.add]
# sp_17 => add_46
triton_poi_fused_add_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 26
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
    tmp1 = tl.load(in_ptr1 + (26 + x2 + (104*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (26*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kb/ckbtwtllxdqwh25a74i4zo43kckqcv724lmzkvt5icq33z5xedzn.py
# Source Nodes: [sp_21], Original ATen: [aten.add]
# sp_21 => add_52
triton_poi_fused_add_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 26
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
    tmp1 = tl.load(in_ptr1 + (52 + x2 + (104*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (26*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iw/ciw3oojvz6ofgmcqz4q47ruziahjwpcahvuam7bpm5a35z64m23k.py
# Source Nodes: [cat_64], Original ATen: [aten.cat]
# cat_64 => cat_1
triton_poi_fused_cat_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 208
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 26
    y1 = (yindex // 26)
    tmp0 = tl.load(in_ptr0 + (78 + y0 + (104*x2) + (326144*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (3136*y0) + (326144*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xg/cxgvlbmmrxdzacg5w4byalbwz7dghb6ij6qaip4srofa5uti5gqv.py
# Source Nodes: [out_13, out_14, shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_13 => add_59, add_62, mul_77, mul_83, rsqrt_11, sub_11, var_mean_11
# out_14 => add_63
# shortcut_3 => relu_10
triton_poi_fused__native_batch_norm_legit_functional_add_relu_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xf/cxf2wf2kqjt2wykf4ofp4akicviiqix35hf2buacyk553mrrx2gw.py
# Source Nodes: [out_24], Original ATen: [aten.convolution]
# out_24 => convolution_17
triton_poi_fused_convolution_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1664
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (208*x2) + (652288*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5g/c5ggkth5t2mgrl73s3kmaigyopxiovoie4jryw3l4qg7ic5xr3ht.py
# Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional]
# out_25 => var_mean_17
triton_red_fused__native_batch_norm_legit_functional_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 40768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 208
    x1 = (xindex // 208)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (208*r2) + (26624*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/oa/coamycxrw3q4iecu5434yaskuyn3qgqwa5oep2owq3hx6yvb37wy.py
# Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional]
# out_25 => var_mean_17
triton_red_fused__native_batch_norm_legit_functional_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 416
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
        tmp0 = tl.load(in_ptr0 + (x1 + (208*r2) + (20384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (208*r2) + (20384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (208*r2) + (20384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (208*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (208*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (208*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/db/cdbm5qujsra6viav3cjxhew3o3ibnljowtbldltykh5ui4ups6dc.py
# Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional]
# out_25 => add_93, add_94, add_95, mul_120, mul_121, mul_122, mul_123, mul_124, rsqrt_17, squeeze_52, var_mean_17
triton_per_fused__native_batch_norm_legit_functional_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_35', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 208
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (208*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (208*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (208*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/u7/cu744wo73oxi6jehr2hnjgckob76rc2qzc42sayjoppvsjtmfdl6.py
# Source Nodes: [out_25, out_26], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_25 => add_93, add_96, mul_119, mul_125, rsqrt_17, sub_17, var_mean_17
# out_26 => relu_16
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5218304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 208
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xq/cxqpzgbfihpqg2ih2wh7ggu3z44dctwb5pp26exkflxbinxoiqax.py
# Source Nodes: [sp_40], Original ATen: [aten.convolution]
# sp_40 => convolution_18
triton_poi_fused_convolution_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 52
    y1 = (yindex // 52)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (52*x2) + (40768*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nm/cnmc3gvo67xmxtaf4qb2yi4meh2vlyhhiljdionhaeqi6autame7.py
# Source Nodes: [sp_41], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_41 => var_mean_18
triton_red_fused__native_batch_norm_legit_functional_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2548
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 52
    x1 = (xindex // 52)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (52*r2) + (6656*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/g4/cg44wog6n4t7rjz65plfytx3tzrcrztshb42leed4xa54d2sjqjk.py
# Source Nodes: [sp_41], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_41 => add_100, add_98, add_99, mul_127, mul_128, mul_129, mul_130, mul_131, rsqrt_18, squeeze_55, var_mean_18
triton_per_fused__native_batch_norm_legit_functional_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_39', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (52*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (52*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/s3/cs3knrel5jozvlc5tlz27fxs7w64ghkben3pyg73srk3dcxo6ubj.py
# Source Nodes: [sp_41, sp_42], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# sp_41 => add_101, add_98, mul_126, mul_132, rsqrt_18, sub_18, var_mean_18
# sp_42 => relu_17
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 52
    y1 = (yindex // 52)
    tmp0 = tl.load(in_ptr0 + (y0 + (52*x2) + (40768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (784*y0) + (163072*y1)), tmp14, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (52*x2) + (40768*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sm/csmj7dlty2v3nknegwggtymyp3xkvvcg6zab2i2zgcqfof6vuifl.py
# Source Nodes: [getattr_l__mod___layer2___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer2___0___pool => avg_pool2d_1
triton_poi_fused_avg_pool2d_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 28)
    x2 = xindex % 28
    y0 = yindex % 52
    y1 = (yindex // 52)
    x5 = xindex
    tmp0 = (-1) + (2*x3)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x2)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-11700) + y0 + (416*x2) + (23296*x3) + (652288*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x2
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-11492) + y0 + (416*x2) + (23296*x3) + (652288*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x2)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-11284) + y0 + (416*x2) + (23296*x3) + (652288*y1)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x3
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-52) + y0 + (416*x2) + (23296*x3) + (652288*y1)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (156 + y0 + (416*x2) + (23296*x3) + (652288*y1)), tmp41 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (364 + y0 + (416*x2) + (23296*x3) + (652288*y1)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x3)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (11596 + y0 + (416*x2) + (23296*x3) + (652288*y1)), tmp55 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (11804 + y0 + (416*x2) + (23296*x3) + (652288*y1)), tmp60 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (12012 + y0 + (416*x2) + (23296*x3) + (652288*y1)), tmp65 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1, 1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1, 1], 57, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tl.broadcast_to((-1) + (2*x3), [XBLOCK, YBLOCK])
    tmp80 = tmp79 >= tmp1
    tmp81 = tmp79 < tmp3
    tmp82 = tmp80 & tmp81
    tmp83 = tl.broadcast_to((-1) + (2*x2), [XBLOCK, YBLOCK])
    tmp84 = tmp83 >= tmp1
    tmp85 = tmp83 < tmp3
    tmp86 = tmp84 & tmp85
    tmp87 = tmp82 & tmp86
    tmp88 = tmp87 & tmp78
    tmp89 = 1.0
    tmp90 = tl.full(tmp89.shape, 1.0, tmp89.dtype)
    tmp91 = tl.where(tmp88, tmp89, tmp90)
    tmp92 = tl.full(tmp91.shape, 0.0, tmp91.dtype)
    tmp93 = tl.where(tmp78, tmp91, tmp92)
    tmp94 = tmp14 >= tmp70
    tmp95 = tmp14 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tl.broadcast_to(2*x2, [XBLOCK, YBLOCK])
    tmp99 = tmp98 >= tmp1
    tmp100 = tmp98 < tmp3
    tmp101 = tmp99 & tmp100
    tmp102 = tmp82 & tmp101
    tmp103 = tmp102 & tmp97
    tmp104 = tl.where(tmp103, tmp89, tmp90)
    tmp105 = tl.full(tmp104.shape, 0.0, tmp104.dtype)
    tmp106 = tl.where(tmp97, tmp104, tmp105)
    tmp107 = tmp106 + tmp93
    tmp108 = tmp23 >= tmp70
    tmp109 = tmp23 < tmp72
    tmp110 = tmp108 & tmp109
    tmp111 = tmp74 & tmp110
    tmp112 = tl.broadcast_to(1 + (2*x2), [XBLOCK, YBLOCK])
    tmp113 = tmp112 >= tmp1
    tmp114 = tmp112 < tmp3
    tmp115 = tmp113 & tmp114
    tmp116 = tmp82 & tmp115
    tmp117 = tmp116 & tmp111
    tmp118 = tl.where(tmp117, tmp89, tmp90)
    tmp119 = tl.full(tmp118.shape, 0.0, tmp118.dtype)
    tmp120 = tl.where(tmp111, tmp118, tmp119)
    tmp121 = tmp120 + tmp107
    tmp122 = tmp32 >= tmp70
    tmp123 = tmp32 < tmp72
    tmp124 = tmp122 & tmp123
    tmp125 = tmp124 & tmp77
    tmp126 = tl.broadcast_to(2*x3, [XBLOCK, YBLOCK])
    tmp127 = tmp126 >= tmp1
    tmp128 = tmp126 < tmp3
    tmp129 = tmp127 & tmp128
    tmp130 = tmp129 & tmp86
    tmp131 = tmp130 & tmp125
    tmp132 = tl.where(tmp131, tmp89, tmp90)
    tmp133 = tl.full(tmp132.shape, 0.0, tmp132.dtype)
    tmp134 = tl.where(tmp125, tmp132, tmp133)
    tmp135 = tmp134 + tmp121
    tmp136 = tmp124 & tmp96
    tmp137 = tmp129 & tmp101
    tmp138 = tmp137 & tmp136
    tmp139 = tl.where(tmp138, tmp89, tmp90)
    tmp140 = tl.full(tmp139.shape, 0.0, tmp139.dtype)
    tmp141 = tl.where(tmp136, tmp139, tmp140)
    tmp142 = tmp141 + tmp135
    tmp143 = tmp124 & tmp110
    tmp144 = tmp129 & tmp115
    tmp145 = tmp144 & tmp143
    tmp146 = tl.where(tmp145, tmp89, tmp90)
    tmp147 = tl.full(tmp146.shape, 0.0, tmp146.dtype)
    tmp148 = tl.where(tmp143, tmp146, tmp147)
    tmp149 = tmp148 + tmp142
    tmp150 = tmp51 >= tmp70
    tmp151 = tmp51 < tmp72
    tmp152 = tmp150 & tmp151
    tmp153 = tmp152 & tmp77
    tmp154 = tl.broadcast_to(1 + (2*x3), [XBLOCK, YBLOCK])
    tmp155 = tmp154 >= tmp1
    tmp156 = tmp154 < tmp3
    tmp157 = tmp155 & tmp156
    tmp158 = tmp157 & tmp86
    tmp159 = tmp158 & tmp153
    tmp160 = tl.where(tmp159, tmp89, tmp90)
    tmp161 = tl.full(tmp160.shape, 0.0, tmp160.dtype)
    tmp162 = tl.where(tmp153, tmp160, tmp161)
    tmp163 = tmp162 + tmp149
    tmp164 = tmp152 & tmp96
    tmp165 = tmp157 & tmp101
    tmp166 = tmp165 & tmp164
    tmp167 = tl.where(tmp166, tmp89, tmp90)
    tmp168 = tl.full(tmp167.shape, 0.0, tmp167.dtype)
    tmp169 = tl.where(tmp164, tmp167, tmp168)
    tmp170 = tmp169 + tmp163
    tmp171 = tmp152 & tmp110
    tmp172 = tmp157 & tmp115
    tmp173 = tmp172 & tmp171
    tmp174 = tl.where(tmp173, tmp89, tmp90)
    tmp175 = tl.full(tmp174.shape, 0.0, tmp174.dtype)
    tmp176 = tl.where(tmp171, tmp174, tmp175)
    tmp177 = tmp176 + tmp170
    tmp178 = tmp69 / tmp177
    tl.store(out_ptr0 + (x5 + (784*y0) + (163072*y1)), tmp178, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ae/caenzugnnx7jjlimsey2xyjngg5opiqwo3mkqq2ulvfwbr7diuoy.py
# Source Nodes: [cat_62], Original ATen: [aten.cat]
# cat_62 => cat_3
triton_poi_fused_cat_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1664
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (208*x2) + (163072*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/on/conlecosicomvc6i44yqm5ya44rgc3shql244kyhow6eeebd2hi2.py
# Source Nodes: [out_28], Original ATen: [aten.convolution]
# out_28 => convolution_21
triton_poi_fused_convolution_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (401408*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u5/cu54mvygp5ctw3j6tcebgkdsbuwyecdv7irvmb7dyajovsdsbye4.py
# Source Nodes: [out_29], Original ATen: [aten._native_batch_norm_legit_functional]
# out_29 => var_mean_21
triton_red_fused__native_batch_norm_legit_functional_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/kx/ckxi6lnvloa62tkwpubpcwxyoclwrl2r3zb7at3z4a6ljcgtszpo.py
# Source Nodes: [out_29], Original ATen: [aten._native_batch_norm_legit_functional]
# out_29 => add_113, add_114, add_115, mul_148, mul_149, mul_150, mul_151, mul_152, rsqrt_21, squeeze_64, var_mean_21
triton_per_fused__native_batch_norm_legit_functional_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_45', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/lf/clff5ffmu6rt4dy2yhmhv5k5xb2zu45pxvwmlblkikoz7fjbgssd.py
# Source Nodes: [out_29, out_30, shortcut_5, shortcut_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_29 => add_113, add_116, mul_147, mul_153, rsqrt_21, sub_21, var_mean_21
# out_30 => add_122
# shortcut_5 => add_118, add_121, mul_154, mul_160, rsqrt_22, sub_22, var_mean_22
# shortcut_6 => relu_20
triton_poi_fused__native_batch_norm_legit_functional_add_relu_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_46', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s4/cs457l6i5eqswxrypnga6ddqzon3p5c4dfp36srr4feupftxjpqw.py
# Source Nodes: [out_33], Original ATen: [aten._native_batch_norm_legit_functional]
# out_33 => var_mean_23
triton_red_fused__native_batch_norm_legit_functional_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 208
    x1 = (xindex // 208)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (208*r2) + (26624*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/y5/cy5wkcvta6x52tl774w2dfihy3ivbmr3cxkuisbqfp3qw35u3kid.py
# Source Nodes: [out_33], Original ATen: [aten._native_batch_norm_legit_functional]
# out_33 => add_124, add_125, add_126, mul_162, mul_163, mul_164, mul_165, mul_166, rsqrt_23, squeeze_70, var_mean_23
triton_per_fused__native_batch_norm_legit_functional_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_48', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (208*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (208*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (208*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/nj/cnjmeeo2txh5uzgokz724nepgx44vwgqptentpxuvrls5sdhsgjp.py
# Source Nodes: [out_33, out_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_33 => add_124, add_127, mul_161, mul_167, rsqrt_23, sub_23, var_mean_23
# out_34 => relu_21
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1304576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 208
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ip/cipjzvsjwnon65s6vr2vxstbhnl3673nmkre4ccrz44eapvqc56w.py
# Source Nodes: [sp_56], Original ATen: [aten.add]
# sp_56 => add_133
triton_poi_fused_add_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 52
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
    tmp1 = tl.load(in_ptr1 + (52 + x2 + (208*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (52*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wl/cwlocy7gbdi5ozpluhul6wdna3ly3gy5vn3xpeym5bg4g5emnrd3.py
# Source Nodes: [sp_60], Original ATen: [aten.add]
# sp_60 => add_139
triton_poi_fused_add_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 52
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
    tmp1 = tl.load(in_ptr1 + (104 + x2 + (208*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (52*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ka/ckahftbdmv5aimhg6xcbjeccqg6c5rypm46elelsisl6tbzic5lg.py
# Source Nodes: [cat_61], Original ATen: [aten.cat]
# cat_61 => cat_4
triton_poi_fused_cat_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 52
    y1 = (yindex // 52)
    tmp0 = tl.load(in_ptr0 + (156 + y0 + (208*x2) + (163072*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (784*y0) + (163072*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/re/crehruswxfwlo56pu7474scdubkemaakwbzvso77ezjj4ltuigp2.py
# Source Nodes: [out_37, out_38, shortcut_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_37 => add_146, add_149, mul_189, mul_195, rsqrt_27, sub_27, var_mean_27
# out_38 => add_150
# shortcut_7 => relu_25
triton_poi_fused__native_batch_norm_legit_functional_add_relu_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i4/ci4mfhps2ctb3wh3c7yvoutnqzeg4lwz65msla2zw7cjkekslojl.py
# Source Nodes: [out_56], Original ATen: [aten.convolution]
# out_56 => convolution_38
triton_poi_fused_convolution_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3328
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 416
    y1 = (yindex // 416)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (416*x2) + (326144*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bu/cbupgxeikpnwflzogdu3dufs7nz3bgi3yhyesvz6csc5y7wdt6hf.py
# Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional]
# out_57 => var_mean_38
triton_red_fused__native_batch_norm_legit_functional_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 20384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 416
    x1 = (xindex // 416)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (416*r2) + (53248*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/hs/chs3llefbxuk3d3nv7rvdkfursbhv6ilup62jj2fvnstrpboev5a.py
# Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional]
# out_57 => add_208, add_209, add_210, mul_267, mul_268, mul_269, mul_270, mul_271, rsqrt_38, squeeze_115, var_mean_38
triton_per_fused__native_batch_norm_legit_functional_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_56', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (416*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (416*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (416*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/oe/coej2fv443andpiodjtqufbnopimqrynjusktu4jjurhjcbvwtu2.py
# Source Nodes: [out_57, out_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_57 => add_208, add_211, mul_266, mul_272, rsqrt_38, sub_38, var_mean_38
# out_58 => relu_36
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2609152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 416
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/x2/cx2hpyqrqynfk7i3y5cy7m3ixdoo3nnhdoulxkwsfkkmtipbikum.py
# Source Nodes: [sp_92], Original ATen: [aten.convolution]
# sp_92 => convolution_39
triton_poi_fused_convolution_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (104*x2) + (20384*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/57/c57guuaj6rfzcjeeu66pbqlprggtvmit4a5rs6nieoezefd6x6pv.py
# Source Nodes: [sp_93], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_93 => var_mean_39
triton_red_fused__native_batch_norm_legit_functional_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1352
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 104)
    x0 = xindex % 104
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (104*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lf/clfwtphwd5u6dnhunj2ukq7e5fyom4rkhtltifzv4wjk5s7z4ojd.py
# Source Nodes: [sp_93], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_93 => add_213, add_214, add_215, mul_274, mul_275, mul_276, mul_277, mul_278, rsqrt_39, squeeze_118, var_mean_39
triton_per_fused__native_batch_norm_legit_functional_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_60', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (104*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (104*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 1568.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0006381620931717
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


# kernel path: /tmp/torchinductor_youkaichao/dq/cdq2ytkdno7a6wl2f5khnzkvkas4r4dbeix3igewr6ngye5g2nqi.py
# Source Nodes: [sp_93, sp_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# sp_93 => add_213, add_216, mul_273, mul_279, rsqrt_39, sub_39, var_mean_39
# sp_94 => relu_37
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (y0 + (104*x2) + (20384*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (196*y0) + (81536*y1)), tmp14, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (104*x2) + (20384*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/us/cusr7dfcjcxtekg6mpofaia3aw5g2itfwg3q5liasyysbuv7ewza.py
# Source Nodes: [getattr_l__mod___layer3___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer3___0___pool => avg_pool2d_2
triton_poi_fused_avg_pool2d_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_62', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 14)
    x2 = xindex % 14
    y0 = yindex % 104
    y1 = (yindex // 104)
    x5 = xindex
    tmp0 = (-1) + (2*x3)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x2)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-11752) + y0 + (832*x2) + (23296*x3) + (326144*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x2
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-11336) + y0 + (832*x2) + (23296*x3) + (326144*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x2)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-10920) + y0 + (832*x2) + (23296*x3) + (326144*y1)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x3
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-104) + y0 + (832*x2) + (23296*x3) + (326144*y1)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (312 + y0 + (832*x2) + (23296*x3) + (326144*y1)), tmp41 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (728 + y0 + (832*x2) + (23296*x3) + (326144*y1)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x3)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (11544 + y0 + (832*x2) + (23296*x3) + (326144*y1)), tmp55 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (11960 + y0 + (832*x2) + (23296*x3) + (326144*y1)), tmp60 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (12376 + y0 + (832*x2) + (23296*x3) + (326144*y1)), tmp65 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1, 1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1, 1], 29, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tl.broadcast_to((-1) + (2*x3), [XBLOCK, YBLOCK])
    tmp80 = tmp79 >= tmp1
    tmp81 = tmp79 < tmp3
    tmp82 = tmp80 & tmp81
    tmp83 = tl.broadcast_to((-1) + (2*x2), [XBLOCK, YBLOCK])
    tmp84 = tmp83 >= tmp1
    tmp85 = tmp83 < tmp3
    tmp86 = tmp84 & tmp85
    tmp87 = tmp82 & tmp86
    tmp88 = tmp87 & tmp78
    tmp89 = 1.0
    tmp90 = tl.full(tmp89.shape, 1.0, tmp89.dtype)
    tmp91 = tl.where(tmp88, tmp89, tmp90)
    tmp92 = tl.full(tmp91.shape, 0.0, tmp91.dtype)
    tmp93 = tl.where(tmp78, tmp91, tmp92)
    tmp94 = tmp14 >= tmp70
    tmp95 = tmp14 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tl.broadcast_to(2*x2, [XBLOCK, YBLOCK])
    tmp99 = tmp98 >= tmp1
    tmp100 = tmp98 < tmp3
    tmp101 = tmp99 & tmp100
    tmp102 = tmp82 & tmp101
    tmp103 = tmp102 & tmp97
    tmp104 = tl.where(tmp103, tmp89, tmp90)
    tmp105 = tl.full(tmp104.shape, 0.0, tmp104.dtype)
    tmp106 = tl.where(tmp97, tmp104, tmp105)
    tmp107 = tmp106 + tmp93
    tmp108 = tmp23 >= tmp70
    tmp109 = tmp23 < tmp72
    tmp110 = tmp108 & tmp109
    tmp111 = tmp74 & tmp110
    tmp112 = tl.broadcast_to(1 + (2*x2), [XBLOCK, YBLOCK])
    tmp113 = tmp112 >= tmp1
    tmp114 = tmp112 < tmp3
    tmp115 = tmp113 & tmp114
    tmp116 = tmp82 & tmp115
    tmp117 = tmp116 & tmp111
    tmp118 = tl.where(tmp117, tmp89, tmp90)
    tmp119 = tl.full(tmp118.shape, 0.0, tmp118.dtype)
    tmp120 = tl.where(tmp111, tmp118, tmp119)
    tmp121 = tmp120 + tmp107
    tmp122 = tmp32 >= tmp70
    tmp123 = tmp32 < tmp72
    tmp124 = tmp122 & tmp123
    tmp125 = tmp124 & tmp77
    tmp126 = tl.broadcast_to(2*x3, [XBLOCK, YBLOCK])
    tmp127 = tmp126 >= tmp1
    tmp128 = tmp126 < tmp3
    tmp129 = tmp127 & tmp128
    tmp130 = tmp129 & tmp86
    tmp131 = tmp130 & tmp125
    tmp132 = tl.where(tmp131, tmp89, tmp90)
    tmp133 = tl.full(tmp132.shape, 0.0, tmp132.dtype)
    tmp134 = tl.where(tmp125, tmp132, tmp133)
    tmp135 = tmp134 + tmp121
    tmp136 = tmp124 & tmp96
    tmp137 = tmp129 & tmp101
    tmp138 = tmp137 & tmp136
    tmp139 = tl.where(tmp138, tmp89, tmp90)
    tmp140 = tl.full(tmp139.shape, 0.0, tmp139.dtype)
    tmp141 = tl.where(tmp136, tmp139, tmp140)
    tmp142 = tmp141 + tmp135
    tmp143 = tmp124 & tmp110
    tmp144 = tmp129 & tmp115
    tmp145 = tmp144 & tmp143
    tmp146 = tl.where(tmp145, tmp89, tmp90)
    tmp147 = tl.full(tmp146.shape, 0.0, tmp146.dtype)
    tmp148 = tl.where(tmp143, tmp146, tmp147)
    tmp149 = tmp148 + tmp142
    tmp150 = tmp51 >= tmp70
    tmp151 = tmp51 < tmp72
    tmp152 = tmp150 & tmp151
    tmp153 = tmp152 & tmp77
    tmp154 = tl.broadcast_to(1 + (2*x3), [XBLOCK, YBLOCK])
    tmp155 = tmp154 >= tmp1
    tmp156 = tmp154 < tmp3
    tmp157 = tmp155 & tmp156
    tmp158 = tmp157 & tmp86
    tmp159 = tmp158 & tmp153
    tmp160 = tl.where(tmp159, tmp89, tmp90)
    tmp161 = tl.full(tmp160.shape, 0.0, tmp160.dtype)
    tmp162 = tl.where(tmp153, tmp160, tmp161)
    tmp163 = tmp162 + tmp149
    tmp164 = tmp152 & tmp96
    tmp165 = tmp157 & tmp101
    tmp166 = tmp165 & tmp164
    tmp167 = tl.where(tmp166, tmp89, tmp90)
    tmp168 = tl.full(tmp167.shape, 0.0, tmp167.dtype)
    tmp169 = tl.where(tmp164, tmp167, tmp168)
    tmp170 = tmp169 + tmp163
    tmp171 = tmp152 & tmp110
    tmp172 = tmp157 & tmp115
    tmp173 = tmp172 & tmp171
    tmp174 = tl.where(tmp173, tmp89, tmp90)
    tmp175 = tl.full(tmp174.shape, 0.0, tmp174.dtype)
    tmp176 = tl.where(tmp171, tmp174, tmp175)
    tmp177 = tmp176 + tmp170
    tmp178 = tmp69 / tmp177
    tl.store(out_ptr0 + (x5 + (196*y0) + (81536*y1)), tmp178, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kv/ckvauhloamtxzuegekqw34elkbu4cxbmendnpji6pghthmshus4v.py
# Source Nodes: [cat_58], Original ATen: [aten.cat]
# cat_58 => cat_7
triton_poi_fused_cat_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3328
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 416
    y1 = (yindex // 416)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (416*x2) + (81536*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rx/crxppckmxsixehukqssit4fywjr7m7zs3eqyxde2one3cbj7bjfj.py
# Source Nodes: [out_60], Original ATen: [aten.convolution]
# out_60 => convolution_42
triton_poi_fused_convolution_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1024*x2) + (200704*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sz/csz4vsmanv3kzxjr6q6h5jafsjysuu6my4uqu2j2j2iqa6lhz4bj.py
# Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
# out_61 => var_mean_42
triton_red_fused__native_batch_norm_legit_functional_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 13312
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 1024)
    x0 = xindex % 1024
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (1024*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2o/c2oqru7tm5kw3l6bvjgcbrryla4jheumlwkmo56pphqiihbg5bls.py
# Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
# out_61 => add_228, add_229, add_230, mul_295, mul_296, mul_297, mul_298, mul_299, rsqrt_42, squeeze_127, var_mean_42
triton_per_fused__native_batch_norm_legit_functional_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_66', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 1568.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0006381620931717
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


# kernel path: /tmp/torchinductor_youkaichao/hh/chhajb5c7krir3l2bx5wso7ibluqyyxccyhqpicwgux44mts4f32.py
# Source Nodes: [out_61, out_62, shortcut_10, shortcut_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_61 => add_228, add_231, mul_294, mul_300, rsqrt_42, sub_42, var_mean_42
# out_62 => add_237
# shortcut_10 => add_233, add_236, mul_301, mul_307, rsqrt_43, sub_43, var_mean_43
# shortcut_11 => relu_40
triton_poi_fused__native_batch_norm_legit_functional_add_relu_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_67', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/a4/ca434ek5wb4y36vchopu63257ebgvvi53eyeysru4xw7tykykxiz.py
# Source Nodes: [out_65], Original ATen: [aten._native_batch_norm_legit_functional]
# out_65 => var_mean_44
triton_red_fused__native_batch_norm_legit_functional_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5408
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 416)
    x0 = xindex % 416
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (416*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mz/cmz3pn5os5qrgmtdi33i7nwsasavgyvh6q6i575cttxbevuyu7jf.py
# Source Nodes: [out_65], Original ATen: [aten._native_batch_norm_legit_functional]
# out_65 => add_239, add_240, add_241, mul_309, mul_310, mul_311, mul_312, mul_313, rsqrt_44, squeeze_133, var_mean_44
triton_per_fused__native_batch_norm_legit_functional_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_69', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (416*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (416*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (416*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 1568.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0006381620931717
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


# kernel path: /tmp/torchinductor_youkaichao/x5/cx5odygdwdxosnlolo5kwyhzqlwqrns6y4wh2zu7tpuhcr5km27s.py
# Source Nodes: [out_65, out_66], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_65 => add_239, add_242, mul_308, mul_314, rsqrt_44, sub_44, var_mean_44
# out_66 => relu_41
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 652288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 416
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2p/c2pcas6jjz72dhtv6llrnalpbbfzig3kiduokurfowo6eap2a75v.py
# Source Nodes: [sp_108], Original ATen: [aten.add]
# sp_108 => add_248
triton_poi_fused_add_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 104
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
    tmp1 = tl.load(in_ptr1 + (104 + x2 + (416*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (104*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kp/ckpke7pujt7bh6xm4wnbuylh2hi5hq2iez6aab5n4xgz3igkdivn.py
# Source Nodes: [sp_112], Original ATen: [aten.add]
# sp_112 => add_254
triton_poi_fused_add_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 104
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
    tmp1 = tl.load(in_ptr1 + (208 + x2 + (416*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (104*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xw/cxwr5kl2lh7o5e5o6a5m34upitbfoaldmigcx6vmiy7zqf6dkdsv.py
# Source Nodes: [cat_57], Original ATen: [aten.cat]
# cat_57 => cat_8
triton_poi_fused_cat_73 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (312 + y0 + (416*x2) + (81536*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (196*y0) + (81536*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g2/cg2usyt5vb57euu3sivalew4muzf5yohg3zqbd6hecebjyst5yhz.py
# Source Nodes: [out_69, out_70, shortcut_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_69 => add_261, add_264, mul_336, mul_342, rsqrt_48, sub_48, var_mean_48
# out_70 => add_265
# shortcut_12 => relu_45
triton_poi_fused__native_batch_norm_legit_functional_add_relu_74 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2s/c2somj23q3zckywtsi2badirf7sv5lbxtqxviv3rcfbfhybax5no.py
# Source Nodes: [out_240], Original ATen: [aten.convolution]
# out_240 => convolution_154
triton_poi_fused_convolution_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6656
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 832
    y1 = (yindex // 832)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (832*x2) + (163072*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qf/cqfmjv2owfhfpknqgth5rwe3h5efhesk2dv2sdlfaaf4nypreq7c.py
# Source Nodes: [out_241], Original ATen: [aten._native_batch_norm_legit_functional]
# out_241 => var_mean_154
triton_red_fused__native_batch_norm_legit_functional_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10816
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 832)
    x0 = xindex % 832
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (832*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c6/cc6ydnuli7yydyhq3pfn3iazbqpljlqp6oz4gxf7zdcvxyye6qns.py
# Source Nodes: [out_241], Original ATen: [aten._native_batch_norm_legit_functional]
# out_241 => add_855, add_856, add_857, mul_1079, mul_1080, mul_1081, mul_1082, mul_1083, rsqrt_154, squeeze_463, var_mean_154
triton_per_fused__native_batch_norm_legit_functional_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_77', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (832*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (832*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (832*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 1568.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0006381620931717
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


# kernel path: /tmp/torchinductor_youkaichao/vq/cvqiai56aardbsid3c2e27qaegbz65khw6l72xgsv554dgde6b5x.py
# Source Nodes: [out_241, out_242], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_241 => add_855, add_858, mul_1078, mul_1084, rsqrt_154, sub_154, var_mean_154
# out_242 => relu_151
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1304576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 832
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yk/cykalrnrkukcyxeqfo4yrcoq2x4jgn47a6gz4t4q5g3dhnc7qjog.py
# Source Nodes: [sp_391], Original ATen: [aten.convolution]
# sp_391 => convolution_155
triton_poi_fused_convolution_79 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1664
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (208*x2) + (10192*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nl/cnlxhganikftfe2gd6osgtdd3bnearf22jbjwzqnx2f54ftprxrn.py
# Source Nodes: [sp_392], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_392 => var_mean_155
triton_red_fused__native_batch_norm_legit_functional_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 208
    x1 = (xindex // 208)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (208*r2) + (20384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/i7/ci7yr3ye67havqxejphifa6ubahow5vhxpg2xiglqmkoxkzgxoa3.py
# Source Nodes: [sp_392], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_392 => add_860, add_861, add_862, mul_1086, mul_1087, mul_1088, mul_1089, mul_1090, rsqrt_155, squeeze_466, var_mean_155
triton_per_fused__native_batch_norm_legit_functional_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_81', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (208*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (208*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 392.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0025575447570332
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


# kernel path: /tmp/torchinductor_youkaichao/vj/cvjk5zz2brd42mi3jacjzmnpikgmv53xpvwrdiezzcmn3i2pudff.py
# Source Nodes: [sp_392, sp_393], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# sp_392 => add_860, add_863, mul_1085, mul_1091, rsqrt_155, sub_155, var_mean_155
# sp_393 => relu_152
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_82 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1664
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    tmp0 = tl.load(in_ptr0 + (y0 + (208*x2) + (10192*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (49*y0) + (40768*y1)), tmp14, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (208*x2) + (10192*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fq/cfqej7amcxcnjqouf6zfckvdvvumdva4khq2ihqnomjfbj2oofiw.py
# Source Nodes: [getattr_l__mod___layer4___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer4___0___pool => avg_pool2d_3
triton_poi_fused_avg_pool2d_83 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_83', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1664
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 7)
    x2 = xindex % 7
    y0 = yindex % 208
    y1 = (yindex // 208)
    x5 = xindex
    tmp0 = (-1) + (2*x3)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x2)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-11856) + y0 + (1664*x2) + (23296*x3) + (163072*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x2
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-11024) + y0 + (1664*x2) + (23296*x3) + (163072*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x2)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-10192) + y0 + (1664*x2) + (23296*x3) + (163072*y1)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x3
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-208) + y0 + (1664*x2) + (23296*x3) + (163072*y1)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (624 + y0 + (1664*x2) + (23296*x3) + (163072*y1)), tmp41 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1456 + y0 + (1664*x2) + (23296*x3) + (163072*y1)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x3)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (11440 + y0 + (1664*x2) + (23296*x3) + (163072*y1)), tmp55 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (12272 + y0 + (1664*x2) + (23296*x3) + (163072*y1)), tmp60 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (13104 + y0 + (1664*x2) + (23296*x3) + (163072*y1)), tmp65 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1, 1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1, 1], 15, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tl.broadcast_to((-1) + (2*x3), [XBLOCK, YBLOCK])
    tmp80 = tmp79 >= tmp1
    tmp81 = tmp79 < tmp3
    tmp82 = tmp80 & tmp81
    tmp83 = tl.broadcast_to((-1) + (2*x2), [XBLOCK, YBLOCK])
    tmp84 = tmp83 >= tmp1
    tmp85 = tmp83 < tmp3
    tmp86 = tmp84 & tmp85
    tmp87 = tmp82 & tmp86
    tmp88 = tmp87 & tmp78
    tmp89 = 1.0
    tmp90 = tl.full(tmp89.shape, 1.0, tmp89.dtype)
    tmp91 = tl.where(tmp88, tmp89, tmp90)
    tmp92 = tl.full(tmp91.shape, 0.0, tmp91.dtype)
    tmp93 = tl.where(tmp78, tmp91, tmp92)
    tmp94 = tmp14 >= tmp70
    tmp95 = tmp14 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tl.broadcast_to(2*x2, [XBLOCK, YBLOCK])
    tmp99 = tmp98 >= tmp1
    tmp100 = tmp98 < tmp3
    tmp101 = tmp99 & tmp100
    tmp102 = tmp82 & tmp101
    tmp103 = tmp102 & tmp97
    tmp104 = tl.where(tmp103, tmp89, tmp90)
    tmp105 = tl.full(tmp104.shape, 0.0, tmp104.dtype)
    tmp106 = tl.where(tmp97, tmp104, tmp105)
    tmp107 = tmp106 + tmp93
    tmp108 = tmp23 >= tmp70
    tmp109 = tmp23 < tmp72
    tmp110 = tmp108 & tmp109
    tmp111 = tmp74 & tmp110
    tmp112 = tl.broadcast_to(1 + (2*x2), [XBLOCK, YBLOCK])
    tmp113 = tmp112 >= tmp1
    tmp114 = tmp112 < tmp3
    tmp115 = tmp113 & tmp114
    tmp116 = tmp82 & tmp115
    tmp117 = tmp116 & tmp111
    tmp118 = tl.where(tmp117, tmp89, tmp90)
    tmp119 = tl.full(tmp118.shape, 0.0, tmp118.dtype)
    tmp120 = tl.where(tmp111, tmp118, tmp119)
    tmp121 = tmp120 + tmp107
    tmp122 = tmp32 >= tmp70
    tmp123 = tmp32 < tmp72
    tmp124 = tmp122 & tmp123
    tmp125 = tmp124 & tmp77
    tmp126 = tl.broadcast_to(2*x3, [XBLOCK, YBLOCK])
    tmp127 = tmp126 >= tmp1
    tmp128 = tmp126 < tmp3
    tmp129 = tmp127 & tmp128
    tmp130 = tmp129 & tmp86
    tmp131 = tmp130 & tmp125
    tmp132 = tl.where(tmp131, tmp89, tmp90)
    tmp133 = tl.full(tmp132.shape, 0.0, tmp132.dtype)
    tmp134 = tl.where(tmp125, tmp132, tmp133)
    tmp135 = tmp134 + tmp121
    tmp136 = tmp124 & tmp96
    tmp137 = tmp129 & tmp101
    tmp138 = tmp137 & tmp136
    tmp139 = tl.where(tmp138, tmp89, tmp90)
    tmp140 = tl.full(tmp139.shape, 0.0, tmp139.dtype)
    tmp141 = tl.where(tmp136, tmp139, tmp140)
    tmp142 = tmp141 + tmp135
    tmp143 = tmp124 & tmp110
    tmp144 = tmp129 & tmp115
    tmp145 = tmp144 & tmp143
    tmp146 = tl.where(tmp145, tmp89, tmp90)
    tmp147 = tl.full(tmp146.shape, 0.0, tmp146.dtype)
    tmp148 = tl.where(tmp143, tmp146, tmp147)
    tmp149 = tmp148 + tmp142
    tmp150 = tmp51 >= tmp70
    tmp151 = tmp51 < tmp72
    tmp152 = tmp150 & tmp151
    tmp153 = tmp152 & tmp77
    tmp154 = tl.broadcast_to(1 + (2*x3), [XBLOCK, YBLOCK])
    tmp155 = tmp154 >= tmp1
    tmp156 = tmp154 < tmp3
    tmp157 = tmp155 & tmp156
    tmp158 = tmp157 & tmp86
    tmp159 = tmp158 & tmp153
    tmp160 = tl.where(tmp159, tmp89, tmp90)
    tmp161 = tl.full(tmp160.shape, 0.0, tmp160.dtype)
    tmp162 = tl.where(tmp153, tmp160, tmp161)
    tmp163 = tmp162 + tmp149
    tmp164 = tmp152 & tmp96
    tmp165 = tmp157 & tmp101
    tmp166 = tmp165 & tmp164
    tmp167 = tl.where(tmp166, tmp89, tmp90)
    tmp168 = tl.full(tmp167.shape, 0.0, tmp167.dtype)
    tmp169 = tl.where(tmp164, tmp167, tmp168)
    tmp170 = tmp169 + tmp163
    tmp171 = tmp152 & tmp110
    tmp172 = tmp157 & tmp115
    tmp173 = tmp172 & tmp171
    tmp174 = tl.where(tmp173, tmp89, tmp90)
    tmp175 = tl.full(tmp174.shape, 0.0, tmp174.dtype)
    tmp176 = tl.where(tmp171, tmp174, tmp175)
    tmp177 = tmp176 + tmp170
    tmp178 = tmp69 / tmp177
    tl.store(out_ptr0 + (x5 + (49*y0) + (40768*y1)), tmp178, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rv/crvzaogmgi66scsf6pmcqtyuc3ov3kry4ui7kz2uimc6aoesy5ee.py
# Source Nodes: [cat_35], Original ATen: [aten.cat]
# cat_35 => cat_30
triton_poi_fused_cat_84 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6656
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 832
    y1 = (yindex // 832)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (832*x2) + (40768*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yq/cyq47auty26l66ayfwkh3yjgjlyzqe2nv6e7xuuzos5ynprknfak.py
# Source Nodes: [out_244], Original ATen: [aten.convolution]
# out_244 => convolution_158
triton_poi_fused_convolution_85 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_85', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 2048
    y1 = (yindex // 2048)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (2048*x2) + (100352*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i6/ci6ujlirbciv4nyfnv6mpwie4cfxm4u4n3imyrbcpwu7q6q4b5bs.py
# Source Nodes: [out_245], Original ATen: [aten._native_batch_norm_legit_functional]
# out_245 => var_mean_158
triton_red_fused__native_batch_norm_legit_functional_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_86', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2) + (200704*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/x7/cx76n2fsxublg4yzxdr4tvbxwkjriuqxqipw55tbk6wdac224hlw.py
# Source Nodes: [out_245], Original ATen: [aten._native_batch_norm_legit_functional]
# out_245 => add_875, add_876, add_877, mul_1107, mul_1108, mul_1109, mul_1110, mul_1111, rsqrt_158, squeeze_475, var_mean_158
triton_per_fused__native_batch_norm_legit_functional_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_87', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (2048*r1)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (2048*r1)), rmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp3, 0)
    tmp8 = tl.where(rmask, tmp4, 0)
    tmp9 = tl.where(rmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 392.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0025575447570332
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, None)
    tl.store(out_ptr4 + (x0), tmp26, None)
    tl.store(out_ptr6 + (x0), tmp32, None)
    tl.store(out_ptr0 + (x0), tmp13, None)
    tl.store(out_ptr1 + (x0), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jm/cjml75s4cf37lwpjpmisb34srzq2bsgjgqkbjwg74cygfddareim.py
# Source Nodes: [out_245, out_246, shortcut_34, shortcut_35], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_245 => add_875, add_878, mul_1106, mul_1112, rsqrt_158, sub_158, var_mean_158
# out_246 => add_884
# shortcut_34 => add_880, add_883, mul_1113, mul_1119, rsqrt_159, sub_159, var_mean_159
# shortcut_35 => relu_155
triton_poi_fused__native_batch_norm_legit_functional_add_relu_88 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_88', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ba/cbabtiwqgvdkdzevapphungu4xsf5yk4dfubpnw7y6aqwhzsjjma.py
# Source Nodes: [out_249], Original ATen: [aten._native_batch_norm_legit_functional]
# out_249 => var_mean_160
triton_red_fused__native_batch_norm_legit_functional_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 832
    x1 = (xindex // 832)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (832*r2) + (81536*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/tk/ctkhuxvigfptzejy6fbjavke4ym5ujn5y6yvut2twgsxssawqeq2.py
# Source Nodes: [out_249], Original ATen: [aten._native_batch_norm_legit_functional]
# out_249 => add_886, add_887, add_888, mul_1121, mul_1122, mul_1123, mul_1124, mul_1125, rsqrt_160, squeeze_481, var_mean_160
triton_per_fused__native_batch_norm_legit_functional_90 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_90', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (832*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (832*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 392.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0025575447570332
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


# kernel path: /tmp/torchinductor_youkaichao/h4/ch4xb4snoollnskb4nyv4jbivnbgmmlj4niooz5xzt2gqvx2x7hq.py
# Source Nodes: [out_249, out_250], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_249 => add_886, add_889, mul_1120, mul_1126, rsqrt_160, sub_160, var_mean_160
# out_250 => relu_156
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_91 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_91', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 326144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 832
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bj/cbjg63m2pzgy4u6ttmsiqsffx2gbnfwqwoz4vqdygwjv7mwd3i57.py
# Source Nodes: [sp_407], Original ATen: [aten.add]
# sp_407 => add_895
triton_poi_fused_add_92 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_92', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 208
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
    tmp1 = tl.load(in_ptr1 + (208 + x2 + (832*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (208*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hx/chx2iazus2434buinvpgnqzpqm52cqsn6c4ymqgvnxn5jrgyiapw.py
# Source Nodes: [sp_411], Original ATen: [aten.add]
# sp_411 => add_901
triton_poi_fused_add_93 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_93', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 208
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
    tmp1 = tl.load(in_ptr1 + (416 + x2 + (832*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (208*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wd/cwdlobc4oxcldqowftpdkqzbtixuqzjwo7ks7la2nmz236em42th.py
# Source Nodes: [cat_34], Original ATen: [aten.cat]
# cat_34 => cat_31
triton_poi_fused_cat_94 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_94', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1664
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    tmp0 = tl.load(in_ptr0 + (624 + y0 + (832*x2) + (40768*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (49*y0) + (40768*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/px/cpxbtmwcesrh3f7ye7y7sovkakl7rxj67qmq5iwm4mm6vs3bt4th.py
# Source Nodes: [out_253, out_254, shortcut_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_253 => add_908, add_911, mul_1148, mul_1154, rsqrt_164, sub_164, var_mean_164
# out_254 => add_912
# shortcut_36 => relu_160
triton_poi_fused__native_batch_norm_legit_functional_add_relu_95 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_95', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6k/c6k3ripjjb5mgwxhtfgaovthb2rvfj7mdpszeloigwxht3t6wcaz.py
# Source Nodes: [out_261, out_262, x_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
# out_261 => add_936, add_939, mul_1183, mul_1189, rsqrt_169, sub_169, var_mean_169
# out_262 => add_940
# x_8 => relu_165
triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_96 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_96', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tmp17 = 0.0
    tmp18 = tmp16 <= tmp17
    tl.store(out_ptr0 + (x2), tmp16, None)
    tl.store(out_ptr1 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hg/chgtvnjrx5kdojbm5kuncrhmjds7bjy43z6pbk4ryp3ieusazyre.py
# Source Nodes: [x_11, x_9], Original ATen: [aten.mean, aten.view]
# x_11 => view
# x_9 => mean
triton_per_fused_mean_view_97 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_view_97', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2) + (100352*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sv/csvyutcmvxfhwfb2qwbvjynnapduafxw42l3mzp2oreu62jigisf.py
# Source Nodes: [x_1], Original ATen: [aten.add]
# x_1 => add
triton_poi_fused_add_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_98', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_945, primals_946, primals_947, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_957, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_964, primals_965, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, primals_977, primals_978, primals_979, primals_980, primals_981, primals_982, primals_983, primals_984, primals_985, primals_986, primals_987, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_995, primals_996, primals_997, primals_998, primals_999, primals_1000, primals_1001, primals_1002, primals_1003, primals_1004, primals_1005, primals_1006, primals_1007, primals_1008, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1014, primals_1015, primals_1016, primals_1017, primals_1018, primals_1019, primals_1020, primals_1021, primals_1022, primals_1023 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (104, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_5, (104, ), (1, ))
    assert_size_stride(primals_6, (104, ), (1, ))
    assert_size_stride(primals_7, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(primals_8, (26, ), (1, ))
    assert_size_stride(primals_9, (26, ), (1, ))
    assert_size_stride(primals_10, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(primals_11, (26, ), (1, ))
    assert_size_stride(primals_12, (26, ), (1, ))
    assert_size_stride(primals_13, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(primals_14, (26, ), (1, ))
    assert_size_stride(primals_15, (26, ), (1, ))
    assert_size_stride(primals_16, (256, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_21, (256, ), (1, ))
    assert_size_stride(primals_22, (104, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_23, (104, ), (1, ))
    assert_size_stride(primals_24, (104, ), (1, ))
    assert_size_stride(primals_25, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(primals_26, (26, ), (1, ))
    assert_size_stride(primals_27, (26, ), (1, ))
    assert_size_stride(primals_28, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(primals_29, (26, ), (1, ))
    assert_size_stride(primals_30, (26, ), (1, ))
    assert_size_stride(primals_31, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(primals_32, (26, ), (1, ))
    assert_size_stride(primals_33, (26, ), (1, ))
    assert_size_stride(primals_34, (256, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_36, (256, ), (1, ))
    assert_size_stride(primals_37, (104, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_38, (104, ), (1, ))
    assert_size_stride(primals_39, (104, ), (1, ))
    assert_size_stride(primals_40, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(primals_41, (26, ), (1, ))
    assert_size_stride(primals_42, (26, ), (1, ))
    assert_size_stride(primals_43, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(primals_44, (26, ), (1, ))
    assert_size_stride(primals_45, (26, ), (1, ))
    assert_size_stride(primals_46, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(primals_47, (26, ), (1, ))
    assert_size_stride(primals_48, (26, ), (1, ))
    assert_size_stride(primals_49, (256, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_51, (256, ), (1, ))
    assert_size_stride(primals_52, (208, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_53, (208, ), (1, ))
    assert_size_stride(primals_54, (208, ), (1, ))
    assert_size_stride(primals_55, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_56, (52, ), (1, ))
    assert_size_stride(primals_57, (52, ), (1, ))
    assert_size_stride(primals_58, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_59, (52, ), (1, ))
    assert_size_stride(primals_60, (52, ), (1, ))
    assert_size_stride(primals_61, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_62, (52, ), (1, ))
    assert_size_stride(primals_63, (52, ), (1, ))
    assert_size_stride(primals_64, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_66, (512, ), (1, ))
    assert_size_stride(primals_67, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_68, (512, ), (1, ))
    assert_size_stride(primals_69, (512, ), (1, ))
    assert_size_stride(primals_70, (208, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_71, (208, ), (1, ))
    assert_size_stride(primals_72, (208, ), (1, ))
    assert_size_stride(primals_73, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_74, (52, ), (1, ))
    assert_size_stride(primals_75, (52, ), (1, ))
    assert_size_stride(primals_76, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_77, (52, ), (1, ))
    assert_size_stride(primals_78, (52, ), (1, ))
    assert_size_stride(primals_79, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_80, (52, ), (1, ))
    assert_size_stride(primals_81, (52, ), (1, ))
    assert_size_stride(primals_82, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(primals_83, (512, ), (1, ))
    assert_size_stride(primals_84, (512, ), (1, ))
    assert_size_stride(primals_85, (208, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_86, (208, ), (1, ))
    assert_size_stride(primals_87, (208, ), (1, ))
    assert_size_stride(primals_88, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_89, (52, ), (1, ))
    assert_size_stride(primals_90, (52, ), (1, ))
    assert_size_stride(primals_91, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_92, (52, ), (1, ))
    assert_size_stride(primals_93, (52, ), (1, ))
    assert_size_stride(primals_94, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_95, (52, ), (1, ))
    assert_size_stride(primals_96, (52, ), (1, ))
    assert_size_stride(primals_97, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(primals_98, (512, ), (1, ))
    assert_size_stride(primals_99, (512, ), (1, ))
    assert_size_stride(primals_100, (208, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_101, (208, ), (1, ))
    assert_size_stride(primals_102, (208, ), (1, ))
    assert_size_stride(primals_103, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_104, (52, ), (1, ))
    assert_size_stride(primals_105, (52, ), (1, ))
    assert_size_stride(primals_106, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_107, (52, ), (1, ))
    assert_size_stride(primals_108, (52, ), (1, ))
    assert_size_stride(primals_109, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_110, (52, ), (1, ))
    assert_size_stride(primals_111, (52, ), (1, ))
    assert_size_stride(primals_112, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(primals_113, (512, ), (1, ))
    assert_size_stride(primals_114, (512, ), (1, ))
    assert_size_stride(primals_115, (416, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_116, (416, ), (1, ))
    assert_size_stride(primals_117, (416, ), (1, ))
    assert_size_stride(primals_118, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_119, (104, ), (1, ))
    assert_size_stride(primals_120, (104, ), (1, ))
    assert_size_stride(primals_121, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_122, (104, ), (1, ))
    assert_size_stride(primals_123, (104, ), (1, ))
    assert_size_stride(primals_124, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_125, (104, ), (1, ))
    assert_size_stride(primals_126, (104, ), (1, ))
    assert_size_stride(primals_127, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_128, (1024, ), (1, ))
    assert_size_stride(primals_129, (1024, ), (1, ))
    assert_size_stride(primals_130, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_131, (1024, ), (1, ))
    assert_size_stride(primals_132, (1024, ), (1, ))
    assert_size_stride(primals_133, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_134, (416, ), (1, ))
    assert_size_stride(primals_135, (416, ), (1, ))
    assert_size_stride(primals_136, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_137, (104, ), (1, ))
    assert_size_stride(primals_138, (104, ), (1, ))
    assert_size_stride(primals_139, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_140, (104, ), (1, ))
    assert_size_stride(primals_141, (104, ), (1, ))
    assert_size_stride(primals_142, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_143, (104, ), (1, ))
    assert_size_stride(primals_144, (104, ), (1, ))
    assert_size_stride(primals_145, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_146, (1024, ), (1, ))
    assert_size_stride(primals_147, (1024, ), (1, ))
    assert_size_stride(primals_148, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_149, (416, ), (1, ))
    assert_size_stride(primals_150, (416, ), (1, ))
    assert_size_stride(primals_151, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_152, (104, ), (1, ))
    assert_size_stride(primals_153, (104, ), (1, ))
    assert_size_stride(primals_154, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_155, (104, ), (1, ))
    assert_size_stride(primals_156, (104, ), (1, ))
    assert_size_stride(primals_157, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_158, (104, ), (1, ))
    assert_size_stride(primals_159, (104, ), (1, ))
    assert_size_stride(primals_160, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_161, (1024, ), (1, ))
    assert_size_stride(primals_162, (1024, ), (1, ))
    assert_size_stride(primals_163, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_164, (416, ), (1, ))
    assert_size_stride(primals_165, (416, ), (1, ))
    assert_size_stride(primals_166, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_167, (104, ), (1, ))
    assert_size_stride(primals_168, (104, ), (1, ))
    assert_size_stride(primals_169, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_170, (104, ), (1, ))
    assert_size_stride(primals_171, (104, ), (1, ))
    assert_size_stride(primals_172, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_173, (104, ), (1, ))
    assert_size_stride(primals_174, (104, ), (1, ))
    assert_size_stride(primals_175, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_176, (1024, ), (1, ))
    assert_size_stride(primals_177, (1024, ), (1, ))
    assert_size_stride(primals_178, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_179, (416, ), (1, ))
    assert_size_stride(primals_180, (416, ), (1, ))
    assert_size_stride(primals_181, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_182, (104, ), (1, ))
    assert_size_stride(primals_183, (104, ), (1, ))
    assert_size_stride(primals_184, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_185, (104, ), (1, ))
    assert_size_stride(primals_186, (104, ), (1, ))
    assert_size_stride(primals_187, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_188, (104, ), (1, ))
    assert_size_stride(primals_189, (104, ), (1, ))
    assert_size_stride(primals_190, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_191, (1024, ), (1, ))
    assert_size_stride(primals_192, (1024, ), (1, ))
    assert_size_stride(primals_193, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_194, (416, ), (1, ))
    assert_size_stride(primals_195, (416, ), (1, ))
    assert_size_stride(primals_196, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_197, (104, ), (1, ))
    assert_size_stride(primals_198, (104, ), (1, ))
    assert_size_stride(primals_199, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_200, (104, ), (1, ))
    assert_size_stride(primals_201, (104, ), (1, ))
    assert_size_stride(primals_202, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_203, (104, ), (1, ))
    assert_size_stride(primals_204, (104, ), (1, ))
    assert_size_stride(primals_205, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_206, (1024, ), (1, ))
    assert_size_stride(primals_207, (1024, ), (1, ))
    assert_size_stride(primals_208, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_209, (416, ), (1, ))
    assert_size_stride(primals_210, (416, ), (1, ))
    assert_size_stride(primals_211, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_212, (104, ), (1, ))
    assert_size_stride(primals_213, (104, ), (1, ))
    assert_size_stride(primals_214, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_215, (104, ), (1, ))
    assert_size_stride(primals_216, (104, ), (1, ))
    assert_size_stride(primals_217, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_218, (104, ), (1, ))
    assert_size_stride(primals_219, (104, ), (1, ))
    assert_size_stride(primals_220, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_221, (1024, ), (1, ))
    assert_size_stride(primals_222, (1024, ), (1, ))
    assert_size_stride(primals_223, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_224, (416, ), (1, ))
    assert_size_stride(primals_225, (416, ), (1, ))
    assert_size_stride(primals_226, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_227, (104, ), (1, ))
    assert_size_stride(primals_228, (104, ), (1, ))
    assert_size_stride(primals_229, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_230, (104, ), (1, ))
    assert_size_stride(primals_231, (104, ), (1, ))
    assert_size_stride(primals_232, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_233, (104, ), (1, ))
    assert_size_stride(primals_234, (104, ), (1, ))
    assert_size_stride(primals_235, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_236, (1024, ), (1, ))
    assert_size_stride(primals_237, (1024, ), (1, ))
    assert_size_stride(primals_238, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_239, (416, ), (1, ))
    assert_size_stride(primals_240, (416, ), (1, ))
    assert_size_stride(primals_241, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_242, (104, ), (1, ))
    assert_size_stride(primals_243, (104, ), (1, ))
    assert_size_stride(primals_244, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_245, (104, ), (1, ))
    assert_size_stride(primals_246, (104, ), (1, ))
    assert_size_stride(primals_247, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_248, (104, ), (1, ))
    assert_size_stride(primals_249, (104, ), (1, ))
    assert_size_stride(primals_250, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_251, (1024, ), (1, ))
    assert_size_stride(primals_252, (1024, ), (1, ))
    assert_size_stride(primals_253, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_254, (416, ), (1, ))
    assert_size_stride(primals_255, (416, ), (1, ))
    assert_size_stride(primals_256, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_257, (104, ), (1, ))
    assert_size_stride(primals_258, (104, ), (1, ))
    assert_size_stride(primals_259, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_260, (104, ), (1, ))
    assert_size_stride(primals_261, (104, ), (1, ))
    assert_size_stride(primals_262, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_263, (104, ), (1, ))
    assert_size_stride(primals_264, (104, ), (1, ))
    assert_size_stride(primals_265, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_266, (1024, ), (1, ))
    assert_size_stride(primals_267, (1024, ), (1, ))
    assert_size_stride(primals_268, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_269, (416, ), (1, ))
    assert_size_stride(primals_270, (416, ), (1, ))
    assert_size_stride(primals_271, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_272, (104, ), (1, ))
    assert_size_stride(primals_273, (104, ), (1, ))
    assert_size_stride(primals_274, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_275, (104, ), (1, ))
    assert_size_stride(primals_276, (104, ), (1, ))
    assert_size_stride(primals_277, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_278, (104, ), (1, ))
    assert_size_stride(primals_279, (104, ), (1, ))
    assert_size_stride(primals_280, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_281, (1024, ), (1, ))
    assert_size_stride(primals_282, (1024, ), (1, ))
    assert_size_stride(primals_283, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_284, (416, ), (1, ))
    assert_size_stride(primals_285, (416, ), (1, ))
    assert_size_stride(primals_286, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_287, (104, ), (1, ))
    assert_size_stride(primals_288, (104, ), (1, ))
    assert_size_stride(primals_289, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_290, (104, ), (1, ))
    assert_size_stride(primals_291, (104, ), (1, ))
    assert_size_stride(primals_292, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_293, (104, ), (1, ))
    assert_size_stride(primals_294, (104, ), (1, ))
    assert_size_stride(primals_295, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_296, (1024, ), (1, ))
    assert_size_stride(primals_297, (1024, ), (1, ))
    assert_size_stride(primals_298, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_299, (416, ), (1, ))
    assert_size_stride(primals_300, (416, ), (1, ))
    assert_size_stride(primals_301, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_302, (104, ), (1, ))
    assert_size_stride(primals_303, (104, ), (1, ))
    assert_size_stride(primals_304, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_305, (104, ), (1, ))
    assert_size_stride(primals_306, (104, ), (1, ))
    assert_size_stride(primals_307, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_308, (104, ), (1, ))
    assert_size_stride(primals_309, (104, ), (1, ))
    assert_size_stride(primals_310, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_311, (1024, ), (1, ))
    assert_size_stride(primals_312, (1024, ), (1, ))
    assert_size_stride(primals_313, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_314, (416, ), (1, ))
    assert_size_stride(primals_315, (416, ), (1, ))
    assert_size_stride(primals_316, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_317, (104, ), (1, ))
    assert_size_stride(primals_318, (104, ), (1, ))
    assert_size_stride(primals_319, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_320, (104, ), (1, ))
    assert_size_stride(primals_321, (104, ), (1, ))
    assert_size_stride(primals_322, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_323, (104, ), (1, ))
    assert_size_stride(primals_324, (104, ), (1, ))
    assert_size_stride(primals_325, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_326, (1024, ), (1, ))
    assert_size_stride(primals_327, (1024, ), (1, ))
    assert_size_stride(primals_328, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_329, (416, ), (1, ))
    assert_size_stride(primals_330, (416, ), (1, ))
    assert_size_stride(primals_331, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_332, (104, ), (1, ))
    assert_size_stride(primals_333, (104, ), (1, ))
    assert_size_stride(primals_334, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_335, (104, ), (1, ))
    assert_size_stride(primals_336, (104, ), (1, ))
    assert_size_stride(primals_337, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_338, (104, ), (1, ))
    assert_size_stride(primals_339, (104, ), (1, ))
    assert_size_stride(primals_340, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_341, (1024, ), (1, ))
    assert_size_stride(primals_342, (1024, ), (1, ))
    assert_size_stride(primals_343, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_344, (416, ), (1, ))
    assert_size_stride(primals_345, (416, ), (1, ))
    assert_size_stride(primals_346, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_347, (104, ), (1, ))
    assert_size_stride(primals_348, (104, ), (1, ))
    assert_size_stride(primals_349, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_350, (104, ), (1, ))
    assert_size_stride(primals_351, (104, ), (1, ))
    assert_size_stride(primals_352, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_353, (104, ), (1, ))
    assert_size_stride(primals_354, (104, ), (1, ))
    assert_size_stride(primals_355, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_356, (1024, ), (1, ))
    assert_size_stride(primals_357, (1024, ), (1, ))
    assert_size_stride(primals_358, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_359, (416, ), (1, ))
    assert_size_stride(primals_360, (416, ), (1, ))
    assert_size_stride(primals_361, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_362, (104, ), (1, ))
    assert_size_stride(primals_363, (104, ), (1, ))
    assert_size_stride(primals_364, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_365, (104, ), (1, ))
    assert_size_stride(primals_366, (104, ), (1, ))
    assert_size_stride(primals_367, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_368, (104, ), (1, ))
    assert_size_stride(primals_369, (104, ), (1, ))
    assert_size_stride(primals_370, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_371, (1024, ), (1, ))
    assert_size_stride(primals_372, (1024, ), (1, ))
    assert_size_stride(primals_373, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_374, (416, ), (1, ))
    assert_size_stride(primals_375, (416, ), (1, ))
    assert_size_stride(primals_376, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_377, (104, ), (1, ))
    assert_size_stride(primals_378, (104, ), (1, ))
    assert_size_stride(primals_379, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_380, (104, ), (1, ))
    assert_size_stride(primals_381, (104, ), (1, ))
    assert_size_stride(primals_382, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_383, (104, ), (1, ))
    assert_size_stride(primals_384, (104, ), (1, ))
    assert_size_stride(primals_385, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_386, (1024, ), (1, ))
    assert_size_stride(primals_387, (1024, ), (1, ))
    assert_size_stride(primals_388, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_389, (416, ), (1, ))
    assert_size_stride(primals_390, (416, ), (1, ))
    assert_size_stride(primals_391, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_392, (104, ), (1, ))
    assert_size_stride(primals_393, (104, ), (1, ))
    assert_size_stride(primals_394, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_395, (104, ), (1, ))
    assert_size_stride(primals_396, (104, ), (1, ))
    assert_size_stride(primals_397, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_398, (104, ), (1, ))
    assert_size_stride(primals_399, (104, ), (1, ))
    assert_size_stride(primals_400, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_401, (1024, ), (1, ))
    assert_size_stride(primals_402, (1024, ), (1, ))
    assert_size_stride(primals_403, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_404, (416, ), (1, ))
    assert_size_stride(primals_405, (416, ), (1, ))
    assert_size_stride(primals_406, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_407, (104, ), (1, ))
    assert_size_stride(primals_408, (104, ), (1, ))
    assert_size_stride(primals_409, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_410, (104, ), (1, ))
    assert_size_stride(primals_411, (104, ), (1, ))
    assert_size_stride(primals_412, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_413, (104, ), (1, ))
    assert_size_stride(primals_414, (104, ), (1, ))
    assert_size_stride(primals_415, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_416, (1024, ), (1, ))
    assert_size_stride(primals_417, (1024, ), (1, ))
    assert_size_stride(primals_418, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_419, (416, ), (1, ))
    assert_size_stride(primals_420, (416, ), (1, ))
    assert_size_stride(primals_421, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_422, (104, ), (1, ))
    assert_size_stride(primals_423, (104, ), (1, ))
    assert_size_stride(primals_424, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_425, (104, ), (1, ))
    assert_size_stride(primals_426, (104, ), (1, ))
    assert_size_stride(primals_427, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_428, (104, ), (1, ))
    assert_size_stride(primals_429, (104, ), (1, ))
    assert_size_stride(primals_430, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_431, (1024, ), (1, ))
    assert_size_stride(primals_432, (1024, ), (1, ))
    assert_size_stride(primals_433, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_434, (416, ), (1, ))
    assert_size_stride(primals_435, (416, ), (1, ))
    assert_size_stride(primals_436, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_437, (104, ), (1, ))
    assert_size_stride(primals_438, (104, ), (1, ))
    assert_size_stride(primals_439, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_440, (104, ), (1, ))
    assert_size_stride(primals_441, (104, ), (1, ))
    assert_size_stride(primals_442, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_443, (104, ), (1, ))
    assert_size_stride(primals_444, (104, ), (1, ))
    assert_size_stride(primals_445, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_446, (1024, ), (1, ))
    assert_size_stride(primals_447, (1024, ), (1, ))
    assert_size_stride(primals_448, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_449, (416, ), (1, ))
    assert_size_stride(primals_450, (416, ), (1, ))
    assert_size_stride(primals_451, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_452, (104, ), (1, ))
    assert_size_stride(primals_453, (104, ), (1, ))
    assert_size_stride(primals_454, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_455, (104, ), (1, ))
    assert_size_stride(primals_456, (104, ), (1, ))
    assert_size_stride(primals_457, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_458, (104, ), (1, ))
    assert_size_stride(primals_459, (104, ), (1, ))
    assert_size_stride(primals_460, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_461, (1024, ), (1, ))
    assert_size_stride(primals_462, (1024, ), (1, ))
    assert_size_stride(primals_463, (832, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_464, (832, ), (1, ))
    assert_size_stride(primals_465, (832, ), (1, ))
    assert_size_stride(primals_466, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(primals_467, (208, ), (1, ))
    assert_size_stride(primals_468, (208, ), (1, ))
    assert_size_stride(primals_469, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(primals_470, (208, ), (1, ))
    assert_size_stride(primals_471, (208, ), (1, ))
    assert_size_stride(primals_472, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(primals_473, (208, ), (1, ))
    assert_size_stride(primals_474, (208, ), (1, ))
    assert_size_stride(primals_475, (2048, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(primals_476, (2048, ), (1, ))
    assert_size_stride(primals_477, (2048, ), (1, ))
    assert_size_stride(primals_478, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_479, (2048, ), (1, ))
    assert_size_stride(primals_480, (2048, ), (1, ))
    assert_size_stride(primals_481, (832, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_482, (832, ), (1, ))
    assert_size_stride(primals_483, (832, ), (1, ))
    assert_size_stride(primals_484, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(primals_485, (208, ), (1, ))
    assert_size_stride(primals_486, (208, ), (1, ))
    assert_size_stride(primals_487, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(primals_488, (208, ), (1, ))
    assert_size_stride(primals_489, (208, ), (1, ))
    assert_size_stride(primals_490, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(primals_491, (208, ), (1, ))
    assert_size_stride(primals_492, (208, ), (1, ))
    assert_size_stride(primals_493, (2048, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(primals_494, (2048, ), (1, ))
    assert_size_stride(primals_495, (2048, ), (1, ))
    assert_size_stride(primals_496, (832, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_497, (832, ), (1, ))
    assert_size_stride(primals_498, (832, ), (1, ))
    assert_size_stride(primals_499, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(primals_500, (208, ), (1, ))
    assert_size_stride(primals_501, (208, ), (1, ))
    assert_size_stride(primals_502, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(primals_503, (208, ), (1, ))
    assert_size_stride(primals_504, (208, ), (1, ))
    assert_size_stride(primals_505, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(primals_506, (208, ), (1, ))
    assert_size_stride(primals_507, (208, ), (1, ))
    assert_size_stride(primals_508, (2048, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(primals_509, (2048, ), (1, ))
    assert_size_stride(primals_510, (2048, ), (1, ))
    assert_size_stride(primals_511, (1000, 2048), (2048, 1))
    assert_size_stride(primals_512, (1000, ), (1, ))
    assert_size_stride(primals_513, (64, ), (1, ))
    assert_size_stride(primals_514, (64, ), (1, ))
    assert_size_stride(primals_515, (), ())
    assert_size_stride(primals_516, (104, ), (1, ))
    assert_size_stride(primals_517, (104, ), (1, ))
    assert_size_stride(primals_518, (), ())
    assert_size_stride(primals_519, (26, ), (1, ))
    assert_size_stride(primals_520, (26, ), (1, ))
    assert_size_stride(primals_521, (), ())
    assert_size_stride(primals_522, (26, ), (1, ))
    assert_size_stride(primals_523, (26, ), (1, ))
    assert_size_stride(primals_524, (), ())
    assert_size_stride(primals_525, (26, ), (1, ))
    assert_size_stride(primals_526, (26, ), (1, ))
    assert_size_stride(primals_527, (), ())
    assert_size_stride(primals_528, (256, ), (1, ))
    assert_size_stride(primals_529, (256, ), (1, ))
    assert_size_stride(primals_530, (), ())
    assert_size_stride(primals_531, (256, ), (1, ))
    assert_size_stride(primals_532, (256, ), (1, ))
    assert_size_stride(primals_533, (), ())
    assert_size_stride(primals_534, (104, ), (1, ))
    assert_size_stride(primals_535, (104, ), (1, ))
    assert_size_stride(primals_536, (), ())
    assert_size_stride(primals_537, (26, ), (1, ))
    assert_size_stride(primals_538, (26, ), (1, ))
    assert_size_stride(primals_539, (), ())
    assert_size_stride(primals_540, (26, ), (1, ))
    assert_size_stride(primals_541, (26, ), (1, ))
    assert_size_stride(primals_542, (), ())
    assert_size_stride(primals_543, (26, ), (1, ))
    assert_size_stride(primals_544, (26, ), (1, ))
    assert_size_stride(primals_545, (), ())
    assert_size_stride(primals_546, (256, ), (1, ))
    assert_size_stride(primals_547, (256, ), (1, ))
    assert_size_stride(primals_548, (), ())
    assert_size_stride(primals_549, (104, ), (1, ))
    assert_size_stride(primals_550, (104, ), (1, ))
    assert_size_stride(primals_551, (), ())
    assert_size_stride(primals_552, (26, ), (1, ))
    assert_size_stride(primals_553, (26, ), (1, ))
    assert_size_stride(primals_554, (), ())
    assert_size_stride(primals_555, (26, ), (1, ))
    assert_size_stride(primals_556, (26, ), (1, ))
    assert_size_stride(primals_557, (), ())
    assert_size_stride(primals_558, (26, ), (1, ))
    assert_size_stride(primals_559, (26, ), (1, ))
    assert_size_stride(primals_560, (), ())
    assert_size_stride(primals_561, (256, ), (1, ))
    assert_size_stride(primals_562, (256, ), (1, ))
    assert_size_stride(primals_563, (), ())
    assert_size_stride(primals_564, (208, ), (1, ))
    assert_size_stride(primals_565, (208, ), (1, ))
    assert_size_stride(primals_566, (), ())
    assert_size_stride(primals_567, (52, ), (1, ))
    assert_size_stride(primals_568, (52, ), (1, ))
    assert_size_stride(primals_569, (), ())
    assert_size_stride(primals_570, (52, ), (1, ))
    assert_size_stride(primals_571, (52, ), (1, ))
    assert_size_stride(primals_572, (), ())
    assert_size_stride(primals_573, (52, ), (1, ))
    assert_size_stride(primals_574, (52, ), (1, ))
    assert_size_stride(primals_575, (), ())
    assert_size_stride(primals_576, (512, ), (1, ))
    assert_size_stride(primals_577, (512, ), (1, ))
    assert_size_stride(primals_578, (), ())
    assert_size_stride(primals_579, (512, ), (1, ))
    assert_size_stride(primals_580, (512, ), (1, ))
    assert_size_stride(primals_581, (), ())
    assert_size_stride(primals_582, (208, ), (1, ))
    assert_size_stride(primals_583, (208, ), (1, ))
    assert_size_stride(primals_584, (), ())
    assert_size_stride(primals_585, (52, ), (1, ))
    assert_size_stride(primals_586, (52, ), (1, ))
    assert_size_stride(primals_587, (), ())
    assert_size_stride(primals_588, (52, ), (1, ))
    assert_size_stride(primals_589, (52, ), (1, ))
    assert_size_stride(primals_590, (), ())
    assert_size_stride(primals_591, (52, ), (1, ))
    assert_size_stride(primals_592, (52, ), (1, ))
    assert_size_stride(primals_593, (), ())
    assert_size_stride(primals_594, (512, ), (1, ))
    assert_size_stride(primals_595, (512, ), (1, ))
    assert_size_stride(primals_596, (), ())
    assert_size_stride(primals_597, (208, ), (1, ))
    assert_size_stride(primals_598, (208, ), (1, ))
    assert_size_stride(primals_599, (), ())
    assert_size_stride(primals_600, (52, ), (1, ))
    assert_size_stride(primals_601, (52, ), (1, ))
    assert_size_stride(primals_602, (), ())
    assert_size_stride(primals_603, (52, ), (1, ))
    assert_size_stride(primals_604, (52, ), (1, ))
    assert_size_stride(primals_605, (), ())
    assert_size_stride(primals_606, (52, ), (1, ))
    assert_size_stride(primals_607, (52, ), (1, ))
    assert_size_stride(primals_608, (), ())
    assert_size_stride(primals_609, (512, ), (1, ))
    assert_size_stride(primals_610, (512, ), (1, ))
    assert_size_stride(primals_611, (), ())
    assert_size_stride(primals_612, (208, ), (1, ))
    assert_size_stride(primals_613, (208, ), (1, ))
    assert_size_stride(primals_614, (), ())
    assert_size_stride(primals_615, (52, ), (1, ))
    assert_size_stride(primals_616, (52, ), (1, ))
    assert_size_stride(primals_617, (), ())
    assert_size_stride(primals_618, (52, ), (1, ))
    assert_size_stride(primals_619, (52, ), (1, ))
    assert_size_stride(primals_620, (), ())
    assert_size_stride(primals_621, (52, ), (1, ))
    assert_size_stride(primals_622, (52, ), (1, ))
    assert_size_stride(primals_623, (), ())
    assert_size_stride(primals_624, (512, ), (1, ))
    assert_size_stride(primals_625, (512, ), (1, ))
    assert_size_stride(primals_626, (), ())
    assert_size_stride(primals_627, (416, ), (1, ))
    assert_size_stride(primals_628, (416, ), (1, ))
    assert_size_stride(primals_629, (), ())
    assert_size_stride(primals_630, (104, ), (1, ))
    assert_size_stride(primals_631, (104, ), (1, ))
    assert_size_stride(primals_632, (), ())
    assert_size_stride(primals_633, (104, ), (1, ))
    assert_size_stride(primals_634, (104, ), (1, ))
    assert_size_stride(primals_635, (), ())
    assert_size_stride(primals_636, (104, ), (1, ))
    assert_size_stride(primals_637, (104, ), (1, ))
    assert_size_stride(primals_638, (), ())
    assert_size_stride(primals_639, (1024, ), (1, ))
    assert_size_stride(primals_640, (1024, ), (1, ))
    assert_size_stride(primals_641, (), ())
    assert_size_stride(primals_642, (1024, ), (1, ))
    assert_size_stride(primals_643, (1024, ), (1, ))
    assert_size_stride(primals_644, (), ())
    assert_size_stride(primals_645, (416, ), (1, ))
    assert_size_stride(primals_646, (416, ), (1, ))
    assert_size_stride(primals_647, (), ())
    assert_size_stride(primals_648, (104, ), (1, ))
    assert_size_stride(primals_649, (104, ), (1, ))
    assert_size_stride(primals_650, (), ())
    assert_size_stride(primals_651, (104, ), (1, ))
    assert_size_stride(primals_652, (104, ), (1, ))
    assert_size_stride(primals_653, (), ())
    assert_size_stride(primals_654, (104, ), (1, ))
    assert_size_stride(primals_655, (104, ), (1, ))
    assert_size_stride(primals_656, (), ())
    assert_size_stride(primals_657, (1024, ), (1, ))
    assert_size_stride(primals_658, (1024, ), (1, ))
    assert_size_stride(primals_659, (), ())
    assert_size_stride(primals_660, (416, ), (1, ))
    assert_size_stride(primals_661, (416, ), (1, ))
    assert_size_stride(primals_662, (), ())
    assert_size_stride(primals_663, (104, ), (1, ))
    assert_size_stride(primals_664, (104, ), (1, ))
    assert_size_stride(primals_665, (), ())
    assert_size_stride(primals_666, (104, ), (1, ))
    assert_size_stride(primals_667, (104, ), (1, ))
    assert_size_stride(primals_668, (), ())
    assert_size_stride(primals_669, (104, ), (1, ))
    assert_size_stride(primals_670, (104, ), (1, ))
    assert_size_stride(primals_671, (), ())
    assert_size_stride(primals_672, (1024, ), (1, ))
    assert_size_stride(primals_673, (1024, ), (1, ))
    assert_size_stride(primals_674, (), ())
    assert_size_stride(primals_675, (416, ), (1, ))
    assert_size_stride(primals_676, (416, ), (1, ))
    assert_size_stride(primals_677, (), ())
    assert_size_stride(primals_678, (104, ), (1, ))
    assert_size_stride(primals_679, (104, ), (1, ))
    assert_size_stride(primals_680, (), ())
    assert_size_stride(primals_681, (104, ), (1, ))
    assert_size_stride(primals_682, (104, ), (1, ))
    assert_size_stride(primals_683, (), ())
    assert_size_stride(primals_684, (104, ), (1, ))
    assert_size_stride(primals_685, (104, ), (1, ))
    assert_size_stride(primals_686, (), ())
    assert_size_stride(primals_687, (1024, ), (1, ))
    assert_size_stride(primals_688, (1024, ), (1, ))
    assert_size_stride(primals_689, (), ())
    assert_size_stride(primals_690, (416, ), (1, ))
    assert_size_stride(primals_691, (416, ), (1, ))
    assert_size_stride(primals_692, (), ())
    assert_size_stride(primals_693, (104, ), (1, ))
    assert_size_stride(primals_694, (104, ), (1, ))
    assert_size_stride(primals_695, (), ())
    assert_size_stride(primals_696, (104, ), (1, ))
    assert_size_stride(primals_697, (104, ), (1, ))
    assert_size_stride(primals_698, (), ())
    assert_size_stride(primals_699, (104, ), (1, ))
    assert_size_stride(primals_700, (104, ), (1, ))
    assert_size_stride(primals_701, (), ())
    assert_size_stride(primals_702, (1024, ), (1, ))
    assert_size_stride(primals_703, (1024, ), (1, ))
    assert_size_stride(primals_704, (), ())
    assert_size_stride(primals_705, (416, ), (1, ))
    assert_size_stride(primals_706, (416, ), (1, ))
    assert_size_stride(primals_707, (), ())
    assert_size_stride(primals_708, (104, ), (1, ))
    assert_size_stride(primals_709, (104, ), (1, ))
    assert_size_stride(primals_710, (), ())
    assert_size_stride(primals_711, (104, ), (1, ))
    assert_size_stride(primals_712, (104, ), (1, ))
    assert_size_stride(primals_713, (), ())
    assert_size_stride(primals_714, (104, ), (1, ))
    assert_size_stride(primals_715, (104, ), (1, ))
    assert_size_stride(primals_716, (), ())
    assert_size_stride(primals_717, (1024, ), (1, ))
    assert_size_stride(primals_718, (1024, ), (1, ))
    assert_size_stride(primals_719, (), ())
    assert_size_stride(primals_720, (416, ), (1, ))
    assert_size_stride(primals_721, (416, ), (1, ))
    assert_size_stride(primals_722, (), ())
    assert_size_stride(primals_723, (104, ), (1, ))
    assert_size_stride(primals_724, (104, ), (1, ))
    assert_size_stride(primals_725, (), ())
    assert_size_stride(primals_726, (104, ), (1, ))
    assert_size_stride(primals_727, (104, ), (1, ))
    assert_size_stride(primals_728, (), ())
    assert_size_stride(primals_729, (104, ), (1, ))
    assert_size_stride(primals_730, (104, ), (1, ))
    assert_size_stride(primals_731, (), ())
    assert_size_stride(primals_732, (1024, ), (1, ))
    assert_size_stride(primals_733, (1024, ), (1, ))
    assert_size_stride(primals_734, (), ())
    assert_size_stride(primals_735, (416, ), (1, ))
    assert_size_stride(primals_736, (416, ), (1, ))
    assert_size_stride(primals_737, (), ())
    assert_size_stride(primals_738, (104, ), (1, ))
    assert_size_stride(primals_739, (104, ), (1, ))
    assert_size_stride(primals_740, (), ())
    assert_size_stride(primals_741, (104, ), (1, ))
    assert_size_stride(primals_742, (104, ), (1, ))
    assert_size_stride(primals_743, (), ())
    assert_size_stride(primals_744, (104, ), (1, ))
    assert_size_stride(primals_745, (104, ), (1, ))
    assert_size_stride(primals_746, (), ())
    assert_size_stride(primals_747, (1024, ), (1, ))
    assert_size_stride(primals_748, (1024, ), (1, ))
    assert_size_stride(primals_749, (), ())
    assert_size_stride(primals_750, (416, ), (1, ))
    assert_size_stride(primals_751, (416, ), (1, ))
    assert_size_stride(primals_752, (), ())
    assert_size_stride(primals_753, (104, ), (1, ))
    assert_size_stride(primals_754, (104, ), (1, ))
    assert_size_stride(primals_755, (), ())
    assert_size_stride(primals_756, (104, ), (1, ))
    assert_size_stride(primals_757, (104, ), (1, ))
    assert_size_stride(primals_758, (), ())
    assert_size_stride(primals_759, (104, ), (1, ))
    assert_size_stride(primals_760, (104, ), (1, ))
    assert_size_stride(primals_761, (), ())
    assert_size_stride(primals_762, (1024, ), (1, ))
    assert_size_stride(primals_763, (1024, ), (1, ))
    assert_size_stride(primals_764, (), ())
    assert_size_stride(primals_765, (416, ), (1, ))
    assert_size_stride(primals_766, (416, ), (1, ))
    assert_size_stride(primals_767, (), ())
    assert_size_stride(primals_768, (104, ), (1, ))
    assert_size_stride(primals_769, (104, ), (1, ))
    assert_size_stride(primals_770, (), ())
    assert_size_stride(primals_771, (104, ), (1, ))
    assert_size_stride(primals_772, (104, ), (1, ))
    assert_size_stride(primals_773, (), ())
    assert_size_stride(primals_774, (104, ), (1, ))
    assert_size_stride(primals_775, (104, ), (1, ))
    assert_size_stride(primals_776, (), ())
    assert_size_stride(primals_777, (1024, ), (1, ))
    assert_size_stride(primals_778, (1024, ), (1, ))
    assert_size_stride(primals_779, (), ())
    assert_size_stride(primals_780, (416, ), (1, ))
    assert_size_stride(primals_781, (416, ), (1, ))
    assert_size_stride(primals_782, (), ())
    assert_size_stride(primals_783, (104, ), (1, ))
    assert_size_stride(primals_784, (104, ), (1, ))
    assert_size_stride(primals_785, (), ())
    assert_size_stride(primals_786, (104, ), (1, ))
    assert_size_stride(primals_787, (104, ), (1, ))
    assert_size_stride(primals_788, (), ())
    assert_size_stride(primals_789, (104, ), (1, ))
    assert_size_stride(primals_790, (104, ), (1, ))
    assert_size_stride(primals_791, (), ())
    assert_size_stride(primals_792, (1024, ), (1, ))
    assert_size_stride(primals_793, (1024, ), (1, ))
    assert_size_stride(primals_794, (), ())
    assert_size_stride(primals_795, (416, ), (1, ))
    assert_size_stride(primals_796, (416, ), (1, ))
    assert_size_stride(primals_797, (), ())
    assert_size_stride(primals_798, (104, ), (1, ))
    assert_size_stride(primals_799, (104, ), (1, ))
    assert_size_stride(primals_800, (), ())
    assert_size_stride(primals_801, (104, ), (1, ))
    assert_size_stride(primals_802, (104, ), (1, ))
    assert_size_stride(primals_803, (), ())
    assert_size_stride(primals_804, (104, ), (1, ))
    assert_size_stride(primals_805, (104, ), (1, ))
    assert_size_stride(primals_806, (), ())
    assert_size_stride(primals_807, (1024, ), (1, ))
    assert_size_stride(primals_808, (1024, ), (1, ))
    assert_size_stride(primals_809, (), ())
    assert_size_stride(primals_810, (416, ), (1, ))
    assert_size_stride(primals_811, (416, ), (1, ))
    assert_size_stride(primals_812, (), ())
    assert_size_stride(primals_813, (104, ), (1, ))
    assert_size_stride(primals_814, (104, ), (1, ))
    assert_size_stride(primals_815, (), ())
    assert_size_stride(primals_816, (104, ), (1, ))
    assert_size_stride(primals_817, (104, ), (1, ))
    assert_size_stride(primals_818, (), ())
    assert_size_stride(primals_819, (104, ), (1, ))
    assert_size_stride(primals_820, (104, ), (1, ))
    assert_size_stride(primals_821, (), ())
    assert_size_stride(primals_822, (1024, ), (1, ))
    assert_size_stride(primals_823, (1024, ), (1, ))
    assert_size_stride(primals_824, (), ())
    assert_size_stride(primals_825, (416, ), (1, ))
    assert_size_stride(primals_826, (416, ), (1, ))
    assert_size_stride(primals_827, (), ())
    assert_size_stride(primals_828, (104, ), (1, ))
    assert_size_stride(primals_829, (104, ), (1, ))
    assert_size_stride(primals_830, (), ())
    assert_size_stride(primals_831, (104, ), (1, ))
    assert_size_stride(primals_832, (104, ), (1, ))
    assert_size_stride(primals_833, (), ())
    assert_size_stride(primals_834, (104, ), (1, ))
    assert_size_stride(primals_835, (104, ), (1, ))
    assert_size_stride(primals_836, (), ())
    assert_size_stride(primals_837, (1024, ), (1, ))
    assert_size_stride(primals_838, (1024, ), (1, ))
    assert_size_stride(primals_839, (), ())
    assert_size_stride(primals_840, (416, ), (1, ))
    assert_size_stride(primals_841, (416, ), (1, ))
    assert_size_stride(primals_842, (), ())
    assert_size_stride(primals_843, (104, ), (1, ))
    assert_size_stride(primals_844, (104, ), (1, ))
    assert_size_stride(primals_845, (), ())
    assert_size_stride(primals_846, (104, ), (1, ))
    assert_size_stride(primals_847, (104, ), (1, ))
    assert_size_stride(primals_848, (), ())
    assert_size_stride(primals_849, (104, ), (1, ))
    assert_size_stride(primals_850, (104, ), (1, ))
    assert_size_stride(primals_851, (), ())
    assert_size_stride(primals_852, (1024, ), (1, ))
    assert_size_stride(primals_853, (1024, ), (1, ))
    assert_size_stride(primals_854, (), ())
    assert_size_stride(primals_855, (416, ), (1, ))
    assert_size_stride(primals_856, (416, ), (1, ))
    assert_size_stride(primals_857, (), ())
    assert_size_stride(primals_858, (104, ), (1, ))
    assert_size_stride(primals_859, (104, ), (1, ))
    assert_size_stride(primals_860, (), ())
    assert_size_stride(primals_861, (104, ), (1, ))
    assert_size_stride(primals_862, (104, ), (1, ))
    assert_size_stride(primals_863, (), ())
    assert_size_stride(primals_864, (104, ), (1, ))
    assert_size_stride(primals_865, (104, ), (1, ))
    assert_size_stride(primals_866, (), ())
    assert_size_stride(primals_867, (1024, ), (1, ))
    assert_size_stride(primals_868, (1024, ), (1, ))
    assert_size_stride(primals_869, (), ())
    assert_size_stride(primals_870, (416, ), (1, ))
    assert_size_stride(primals_871, (416, ), (1, ))
    assert_size_stride(primals_872, (), ())
    assert_size_stride(primals_873, (104, ), (1, ))
    assert_size_stride(primals_874, (104, ), (1, ))
    assert_size_stride(primals_875, (), ())
    assert_size_stride(primals_876, (104, ), (1, ))
    assert_size_stride(primals_877, (104, ), (1, ))
    assert_size_stride(primals_878, (), ())
    assert_size_stride(primals_879, (104, ), (1, ))
    assert_size_stride(primals_880, (104, ), (1, ))
    assert_size_stride(primals_881, (), ())
    assert_size_stride(primals_882, (1024, ), (1, ))
    assert_size_stride(primals_883, (1024, ), (1, ))
    assert_size_stride(primals_884, (), ())
    assert_size_stride(primals_885, (416, ), (1, ))
    assert_size_stride(primals_886, (416, ), (1, ))
    assert_size_stride(primals_887, (), ())
    assert_size_stride(primals_888, (104, ), (1, ))
    assert_size_stride(primals_889, (104, ), (1, ))
    assert_size_stride(primals_890, (), ())
    assert_size_stride(primals_891, (104, ), (1, ))
    assert_size_stride(primals_892, (104, ), (1, ))
    assert_size_stride(primals_893, (), ())
    assert_size_stride(primals_894, (104, ), (1, ))
    assert_size_stride(primals_895, (104, ), (1, ))
    assert_size_stride(primals_896, (), ())
    assert_size_stride(primals_897, (1024, ), (1, ))
    assert_size_stride(primals_898, (1024, ), (1, ))
    assert_size_stride(primals_899, (), ())
    assert_size_stride(primals_900, (416, ), (1, ))
    assert_size_stride(primals_901, (416, ), (1, ))
    assert_size_stride(primals_902, (), ())
    assert_size_stride(primals_903, (104, ), (1, ))
    assert_size_stride(primals_904, (104, ), (1, ))
    assert_size_stride(primals_905, (), ())
    assert_size_stride(primals_906, (104, ), (1, ))
    assert_size_stride(primals_907, (104, ), (1, ))
    assert_size_stride(primals_908, (), ())
    assert_size_stride(primals_909, (104, ), (1, ))
    assert_size_stride(primals_910, (104, ), (1, ))
    assert_size_stride(primals_911, (), ())
    assert_size_stride(primals_912, (1024, ), (1, ))
    assert_size_stride(primals_913, (1024, ), (1, ))
    assert_size_stride(primals_914, (), ())
    assert_size_stride(primals_915, (416, ), (1, ))
    assert_size_stride(primals_916, (416, ), (1, ))
    assert_size_stride(primals_917, (), ())
    assert_size_stride(primals_918, (104, ), (1, ))
    assert_size_stride(primals_919, (104, ), (1, ))
    assert_size_stride(primals_920, (), ())
    assert_size_stride(primals_921, (104, ), (1, ))
    assert_size_stride(primals_922, (104, ), (1, ))
    assert_size_stride(primals_923, (), ())
    assert_size_stride(primals_924, (104, ), (1, ))
    assert_size_stride(primals_925, (104, ), (1, ))
    assert_size_stride(primals_926, (), ())
    assert_size_stride(primals_927, (1024, ), (1, ))
    assert_size_stride(primals_928, (1024, ), (1, ))
    assert_size_stride(primals_929, (), ())
    assert_size_stride(primals_930, (416, ), (1, ))
    assert_size_stride(primals_931, (416, ), (1, ))
    assert_size_stride(primals_932, (), ())
    assert_size_stride(primals_933, (104, ), (1, ))
    assert_size_stride(primals_934, (104, ), (1, ))
    assert_size_stride(primals_935, (), ())
    assert_size_stride(primals_936, (104, ), (1, ))
    assert_size_stride(primals_937, (104, ), (1, ))
    assert_size_stride(primals_938, (), ())
    assert_size_stride(primals_939, (104, ), (1, ))
    assert_size_stride(primals_940, (104, ), (1, ))
    assert_size_stride(primals_941, (), ())
    assert_size_stride(primals_942, (1024, ), (1, ))
    assert_size_stride(primals_943, (1024, ), (1, ))
    assert_size_stride(primals_944, (), ())
    assert_size_stride(primals_945, (416, ), (1, ))
    assert_size_stride(primals_946, (416, ), (1, ))
    assert_size_stride(primals_947, (), ())
    assert_size_stride(primals_948, (104, ), (1, ))
    assert_size_stride(primals_949, (104, ), (1, ))
    assert_size_stride(primals_950, (), ())
    assert_size_stride(primals_951, (104, ), (1, ))
    assert_size_stride(primals_952, (104, ), (1, ))
    assert_size_stride(primals_953, (), ())
    assert_size_stride(primals_954, (104, ), (1, ))
    assert_size_stride(primals_955, (104, ), (1, ))
    assert_size_stride(primals_956, (), ())
    assert_size_stride(primals_957, (1024, ), (1, ))
    assert_size_stride(primals_958, (1024, ), (1, ))
    assert_size_stride(primals_959, (), ())
    assert_size_stride(primals_960, (416, ), (1, ))
    assert_size_stride(primals_961, (416, ), (1, ))
    assert_size_stride(primals_962, (), ())
    assert_size_stride(primals_963, (104, ), (1, ))
    assert_size_stride(primals_964, (104, ), (1, ))
    assert_size_stride(primals_965, (), ())
    assert_size_stride(primals_966, (104, ), (1, ))
    assert_size_stride(primals_967, (104, ), (1, ))
    assert_size_stride(primals_968, (), ())
    assert_size_stride(primals_969, (104, ), (1, ))
    assert_size_stride(primals_970, (104, ), (1, ))
    assert_size_stride(primals_971, (), ())
    assert_size_stride(primals_972, (1024, ), (1, ))
    assert_size_stride(primals_973, (1024, ), (1, ))
    assert_size_stride(primals_974, (), ())
    assert_size_stride(primals_975, (832, ), (1, ))
    assert_size_stride(primals_976, (832, ), (1, ))
    assert_size_stride(primals_977, (), ())
    assert_size_stride(primals_978, (208, ), (1, ))
    assert_size_stride(primals_979, (208, ), (1, ))
    assert_size_stride(primals_980, (), ())
    assert_size_stride(primals_981, (208, ), (1, ))
    assert_size_stride(primals_982, (208, ), (1, ))
    assert_size_stride(primals_983, (), ())
    assert_size_stride(primals_984, (208, ), (1, ))
    assert_size_stride(primals_985, (208, ), (1, ))
    assert_size_stride(primals_986, (), ())
    assert_size_stride(primals_987, (2048, ), (1, ))
    assert_size_stride(primals_988, (2048, ), (1, ))
    assert_size_stride(primals_989, (), ())
    assert_size_stride(primals_990, (2048, ), (1, ))
    assert_size_stride(primals_991, (2048, ), (1, ))
    assert_size_stride(primals_992, (), ())
    assert_size_stride(primals_993, (832, ), (1, ))
    assert_size_stride(primals_994, (832, ), (1, ))
    assert_size_stride(primals_995, (), ())
    assert_size_stride(primals_996, (208, ), (1, ))
    assert_size_stride(primals_997, (208, ), (1, ))
    assert_size_stride(primals_998, (), ())
    assert_size_stride(primals_999, (208, ), (1, ))
    assert_size_stride(primals_1000, (208, ), (1, ))
    assert_size_stride(primals_1001, (), ())
    assert_size_stride(primals_1002, (208, ), (1, ))
    assert_size_stride(primals_1003, (208, ), (1, ))
    assert_size_stride(primals_1004, (), ())
    assert_size_stride(primals_1005, (2048, ), (1, ))
    assert_size_stride(primals_1006, (2048, ), (1, ))
    assert_size_stride(primals_1007, (), ())
    assert_size_stride(primals_1008, (832, ), (1, ))
    assert_size_stride(primals_1009, (832, ), (1, ))
    assert_size_stride(primals_1010, (), ())
    assert_size_stride(primals_1011, (208, ), (1, ))
    assert_size_stride(primals_1012, (208, ), (1, ))
    assert_size_stride(primals_1013, (), ())
    assert_size_stride(primals_1014, (208, ), (1, ))
    assert_size_stride(primals_1015, (208, ), (1, ))
    assert_size_stride(primals_1016, (), ())
    assert_size_stride(primals_1017, (208, ), (1, ))
    assert_size_stride(primals_1018, (208, ), (1, ))
    assert_size_stride(primals_1019, (), ())
    assert_size_stride(primals_1020, (2048, ), (1, ))
    assert_size_stride(primals_1021, (2048, ), (1, ))
    assert_size_stride(primals_1022, (), ())
    assert_size_stride(primals_1023, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 192, 49, grid=grid(192, 49), stream=stream0)
        del primals_1
        buf1 = empty_strided((26, 26, 3, 3), (234, 1, 78, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_7, buf1, 676, 9, grid=grid(676, 9), stream=stream0)
        del primals_7
        buf2 = empty_strided((26, 26, 3, 3), (234, 1, 78, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_10, buf2, 676, 9, grid=grid(676, 9), stream=stream0)
        del primals_10
        buf3 = empty_strided((26, 26, 3, 3), (234, 1, 78, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_13, buf3, 676, 9, grid=grid(676, 9), stream=stream0)
        del primals_13
        buf4 = empty_strided((26, 26, 3, 3), (234, 1, 78, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_25, buf4, 676, 9, grid=grid(676, 9), stream=stream0)
        del primals_25
        buf5 = empty_strided((26, 26, 3, 3), (234, 1, 78, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_28, buf5, 676, 9, grid=grid(676, 9), stream=stream0)
        del primals_28
        buf6 = empty_strided((26, 26, 3, 3), (234, 1, 78, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_31, buf6, 676, 9, grid=grid(676, 9), stream=stream0)
        del primals_31
        buf7 = empty_strided((26, 26, 3, 3), (234, 1, 78, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_40, buf7, 676, 9, grid=grid(676, 9), stream=stream0)
        del primals_40
        buf8 = empty_strided((26, 26, 3, 3), (234, 1, 78, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_43, buf8, 676, 9, grid=grid(676, 9), stream=stream0)
        del primals_43
        buf9 = empty_strided((26, 26, 3, 3), (234, 1, 78, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_46, buf9, 676, 9, grid=grid(676, 9), stream=stream0)
        del primals_46
        buf10 = empty_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_55, buf10, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_55
        buf11 = empty_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_58, buf11, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_58
        buf12 = empty_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_61, buf12, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_61
        buf13 = empty_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_73, buf13, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_73
        buf14 = empty_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_76, buf14, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_76
        buf15 = empty_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_79, buf15, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_79
        buf16 = empty_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_88, buf16, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_88
        buf17 = empty_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_91, buf17, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_91
        buf18 = empty_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_94, buf18, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_94
        buf19 = empty_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_103, buf19, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_103
        buf20 = empty_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_106, buf20, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_106
        buf21 = empty_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_109, buf21, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_109
        buf22 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_118, buf22, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_118
        buf23 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_121, buf23, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_121
        buf24 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_124, buf24, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_124
        buf25 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_136, buf25, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_136
        buf26 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_139, buf26, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_139
        buf27 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_142, buf27, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_142
        buf28 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_151, buf28, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_151
        buf29 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_154, buf29, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_154
        buf30 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_157, buf30, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_157
        buf31 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_166, buf31, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_166
        buf32 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_169, buf32, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_169
        buf33 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_172, buf33, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_172
        buf34 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_181, buf34, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_181
        buf35 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_184, buf35, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_184
        buf36 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_187, buf36, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_187
        buf37 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_196, buf37, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_196
        buf38 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_199, buf38, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_199
        buf39 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_202, buf39, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_202
        buf40 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_211, buf40, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_211
        buf41 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_214, buf41, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_214
        buf42 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_217, buf42, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_217
        buf43 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_226, buf43, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_226
        buf44 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_229, buf44, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_229
        buf45 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_232, buf45, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_232
        buf46 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_241, buf46, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_241
        buf47 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_244, buf47, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_244
        buf48 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_247, buf48, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_247
        buf49 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_256, buf49, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_256
        buf50 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_259, buf50, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_259
        buf51 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_262, buf51, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_262
        buf52 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_271, buf52, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_271
        buf53 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_274, buf53, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_274
        buf54 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_277, buf54, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_277
        buf55 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_286, buf55, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_286
        buf56 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_289, buf56, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_289
        buf57 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_292, buf57, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_292
        buf58 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_301, buf58, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_301
        buf59 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_304, buf59, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_304
        buf60 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_307, buf60, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_307
        buf61 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_316, buf61, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_316
        buf62 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_319, buf62, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_319
        buf63 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_322, buf63, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_322
        buf64 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_331, buf64, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_331
        buf65 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_334, buf65, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_334
        buf66 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_337, buf66, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_337
        buf67 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_346, buf67, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_346
        buf68 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_349, buf68, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_349
        buf69 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_352, buf69, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_352
        buf70 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_361, buf70, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_361
        buf71 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_364, buf71, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_364
        buf72 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_367, buf72, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_367
        buf73 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_376, buf73, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_376
        buf74 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_379, buf74, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_379
        buf75 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_382, buf75, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_382
        buf76 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_391, buf76, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_391
        buf77 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_394, buf77, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_394
        buf78 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_397, buf78, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_397
        buf79 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_406, buf79, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_406
        buf80 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_409, buf80, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_409
        buf81 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_412, buf81, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_412
        buf82 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_421, buf82, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_421
        buf83 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_424, buf83, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_424
        buf84 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_427, buf84, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_427
        buf85 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_436, buf85, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_436
        buf86 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_439, buf86, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_439
        buf87 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_442, buf87, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_442
        buf88 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_451, buf88, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_451
        buf89 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_454, buf89, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_454
        buf90 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_457, buf90, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_457
        buf91 = empty_strided((208, 208, 3, 3), (1872, 1, 624, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_466, buf91, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del primals_466
        buf92 = empty_strided((208, 208, 3, 3), (1872, 1, 624, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_469, buf92, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del primals_469
        buf93 = empty_strided((208, 208, 3, 3), (1872, 1, 624, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_472, buf93, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del primals_472
        buf94 = empty_strided((208, 208, 3, 3), (1872, 1, 624, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_484, buf94, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del primals_484
        buf95 = empty_strided((208, 208, 3, 3), (1872, 1, 624, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_487, buf95, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del primals_487
        buf96 = empty_strided((208, 208, 3, 3), (1872, 1, 624, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_490, buf96, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del primals_490
        buf97 = empty_strided((208, 208, 3, 3), (1872, 1, 624, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_499, buf97, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del primals_499
        buf98 = empty_strided((208, 208, 3, 3), (1872, 1, 624, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_502, buf98, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del primals_502
        buf99 = empty_strided((208, 208, 3, 3), (1872, 1, 624, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_505, buf99, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del primals_505
        buf100 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_1023, buf100, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del primals_1023
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, buf0, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (8, 64, 112, 112), (802816, 12544, 112, 1))
        buf102 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(buf101, buf102, 512, 12544, grid=grid(512, 12544), stream=stream0)
        buf103 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        buf104 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        buf105 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf102, buf103, buf104, buf105, 50176, 128, grid=grid(50176), stream=stream0)
        buf106 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        buf107 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        buf108 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf103, buf104, buf105, buf106, buf107, buf108, 448, 112, grid=grid(448), stream=stream0)
        buf109 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf110 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf112 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_9.run(buf106, buf107, buf108, primals_513, primals_514, buf109, buf110, buf112, primals_513, primals_514, 64, 7, grid=grid(64), stream=stream0)
        del buf106
        del buf107
        del buf108
        del primals_513
        del primals_514
        buf113 = reinterpret_tensor(buf101, (8, 64, 112, 112), (802816, 1, 7168, 64), 0); del buf101  # reuse
        # Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_10.run(buf102, buf109, buf110, primals_2, primals_3, buf113, 6422528, grid=grid(6422528), stream=stream0)
        del buf110
        del primals_3
        buf114 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        buf115 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.int64)
        # Source Nodes: [shortcut], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_11.run(buf113, buf114, buf115, 1605632, grid=grid(1605632), stream=stream0)
        # Source Nodes: [out], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf114, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 104, 56, 56), (326144, 3136, 56, 1))
        buf117 = empty_strided((8, 104, 56, 56), (326144, 1, 5824, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [out], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf116, buf117, 832, 3136, grid=grid(832, 3136), stream=stream0)
        buf118 = empty_strided((1, 104, 1, 1, 196), (20384, 1, 20384, 20384, 104), device='cuda', dtype=torch.float32)
        buf119 = empty_strided((1, 104, 1, 1, 196), (20384, 1, 20384, 20384, 104), device='cuda', dtype=torch.float32)
        buf120 = empty_strided((1, 104, 1, 1, 196), (20384, 1, 20384, 20384, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf117, buf118, buf119, buf120, 20384, 128, grid=grid(20384), stream=stream0)
        buf121 = empty_strided((1, 104, 1, 1, 2), (208, 1, 208, 208, 104), device='cuda', dtype=torch.float32)
        buf122 = empty_strided((1, 104, 1, 1, 2), (208, 1, 208, 208, 104), device='cuda', dtype=torch.float32)
        buf123 = empty_strided((1, 104, 1, 1, 2), (208, 1, 208, 208, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf118, buf119, buf120, buf121, buf122, buf123, 208, 98, grid=grid(208), stream=stream0)
        buf124 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf125 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf127 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf121, buf122, buf123, primals_516, primals_517, buf124, buf125, buf127, primals_516, primals_517, 104, 2, grid=grid(104), stream=stream0)
        del primals_516
        del primals_517
        buf128 = reinterpret_tensor(buf116, (8, 104, 56, 56), (326144, 1, 5824, 104), 0); del buf116  # reuse
        buf2149 = empty_strided((8, 104, 56, 56), (326144, 1, 5824, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_16.run(buf117, buf124, buf125, primals_5, primals_6, buf128, buf2149, 2609152, grid=grid(2609152), stream=stream0)
        del primals_6
        # Source Nodes: [sp_1], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(reinterpret_tensor(buf128, (8, 26, 56, 56), (326144, 1, 5824, 104), 0), buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (8, 26, 56, 56), (81536, 3136, 56, 1))
        buf130 = empty_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf129, buf130, 208, 3136, grid=grid(208, 3136), stream=stream0)
        buf131 = empty_strided((1, 26, 1, 1, 196), (5096, 1, 5096, 5096, 26), device='cuda', dtype=torch.float32)
        buf132 = empty_strided((1, 26, 1, 1, 196), (5096, 1, 5096, 5096, 26), device='cuda', dtype=torch.float32)
        buf133 = empty_strided((1, 26, 1, 1, 196), (5096, 1, 5096, 5096, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_2], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf130, buf131, buf132, buf133, 5096, 128, grid=grid(5096), stream=stream0)
        buf134 = empty_strided((1, 26, 1, 1, 2), (52, 1, 52, 52, 26), device='cuda', dtype=torch.float32)
        buf135 = empty_strided((1, 26, 1, 1, 2), (52, 1, 52, 52, 26), device='cuda', dtype=torch.float32)
        buf136 = empty_strided((1, 26, 1, 1, 2), (52, 1, 52, 52, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_2], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf131, buf132, buf133, buf134, buf135, buf136, 52, 98, grid=grid(52), stream=stream0)
        buf137 = empty_strided((1, 26, 1, 1), (26, 1, 26, 26), device='cuda', dtype=torch.float32)
        buf138 = empty_strided((1, 26, 1, 1), (26, 1, 26, 26), device='cuda', dtype=torch.float32)
        buf140 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_2], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf134, buf135, buf136, primals_519, primals_520, buf137, buf138, buf140, primals_519, primals_520, 26, 2, grid=grid(26), stream=stream0)
        del primals_519
        del primals_520
        buf169 = empty((8, 104, 56, 56), device='cuda', dtype=torch.float32)
        buf141 = reinterpret_tensor(buf169, (8, 26, 56, 56), (326144, 3136, 56, 1), 0)  # alias
        buf2148 = empty_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_2, sp_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf130, buf137, buf138, primals_8, primals_9, buf141, buf2148, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del primals_9
        # Source Nodes: [sp_5], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(reinterpret_tensor(buf128, (8, 26, 56, 56), (326144, 1, 5824, 104), 26), buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 26, 56, 56), (81536, 3136, 56, 1))
        buf143 = reinterpret_tensor(buf129, (8, 26, 56, 56), (81536, 1, 1456, 26), 0); del buf129  # reuse
        # Source Nodes: [sp_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf142, buf143, 208, 3136, grid=grid(208, 3136), stream=stream0)
        buf144 = buf133; del buf133  # reuse
        buf145 = buf132; del buf132  # reuse
        buf146 = buf131; del buf131  # reuse
        # Source Nodes: [sp_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf143, buf144, buf145, buf146, 5096, 128, grid=grid(5096), stream=stream0)
        buf147 = buf136; del buf136  # reuse
        buf148 = buf135; del buf135  # reuse
        buf149 = buf134; del buf134  # reuse
        # Source Nodes: [sp_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf144, buf145, buf146, buf147, buf148, buf149, 52, 98, grid=grid(52), stream=stream0)
        buf150 = buf138; del buf138  # reuse
        buf151 = empty_strided((1, 26, 1, 1), (26, 1, 26, 26), device='cuda', dtype=torch.float32)
        buf153 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf147, buf148, buf149, primals_522, primals_523, buf150, buf151, buf153, primals_522, primals_523, 26, 2, grid=grid(26), stream=stream0)
        del primals_522
        del primals_523
        buf154 = reinterpret_tensor(buf169, (8, 26, 56, 56), (326144, 3136, 56, 1), 81536)  # alias
        buf2147 = empty_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_6, sp_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf143, buf150, buf151, primals_11, primals_12, buf154, buf2147, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del primals_12
        # Source Nodes: [sp_9], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(reinterpret_tensor(buf128, (8, 26, 56, 56), (326144, 1, 5824, 104), 52), buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (8, 26, 56, 56), (81536, 3136, 56, 1))
        buf156 = reinterpret_tensor(buf142, (8, 26, 56, 56), (81536, 1, 1456, 26), 0); del buf142  # reuse
        # Source Nodes: [sp_9], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf155, buf156, 208, 3136, grid=grid(208, 3136), stream=stream0)
        buf157 = buf146; del buf146  # reuse
        buf158 = buf145; del buf145  # reuse
        buf159 = buf144; del buf144  # reuse
        # Source Nodes: [sp_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf156, buf157, buf158, buf159, 5096, 128, grid=grid(5096), stream=stream0)
        buf160 = buf149; del buf149  # reuse
        buf161 = buf148; del buf148  # reuse
        buf162 = buf147; del buf147  # reuse
        # Source Nodes: [sp_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf157, buf158, buf159, buf160, buf161, buf162, 52, 98, grid=grid(52), stream=stream0)
        buf163 = buf151; del buf151  # reuse
        buf164 = empty_strided((1, 26, 1, 1), (26, 1, 26, 26), device='cuda', dtype=torch.float32)
        buf166 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf160, buf161, buf162, primals_525, primals_526, buf163, buf164, buf166, primals_525, primals_526, 26, 2, grid=grid(26), stream=stream0)
        del primals_525
        del primals_526
        buf167 = reinterpret_tensor(buf169, (8, 26, 56, 56), (326144, 3136, 56, 1), 163072)  # alias
        buf2146 = empty_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_10, sp_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf156, buf163, buf164, primals_14, primals_15, buf167, buf2146, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del primals_15
        buf168 = reinterpret_tensor(buf169, (8, 26, 56, 56), (326144, 3136, 56, 1), 244608)  # alias
        # Source Nodes: [getattr_l__mod___layer1___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_22.run(buf128, buf168, 208, 3136, grid=grid(208, 3136), stream=stream0)
        buf170 = empty_strided((8, 104, 56, 56), (326144, 1, 5824, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_65], Original ATen: [aten.cat]
        triton_poi_fused_convolution_12.run(buf169, buf170, 832, 3136, grid=grid(832, 3136), stream=stream0)
        del buf141
        del buf154
        del buf167
        del buf168
        # Source Nodes: [out_4], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf172 = empty_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_4], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf171, buf172, 2048, 3136, grid=grid(2048, 3136), stream=stream0)
        buf173 = reinterpret_tensor(buf105, (1, 256, 1, 1, 196), (50176, 1, 50176, 50176, 256), 0); del buf105  # reuse
        buf174 = reinterpret_tensor(buf104, (1, 256, 1, 1, 196), (50176, 1, 50176, 50176, 256), 0); del buf104  # reuse
        buf175 = reinterpret_tensor(buf103, (1, 256, 1, 1, 196), (50176, 1, 50176, 50176, 256), 0); del buf103  # reuse
        # Source Nodes: [out_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf172, buf173, buf174, buf175, 50176, 128, grid=grid(50176), stream=stream0)
        buf176 = empty_strided((1, 256, 1, 1, 2), (512, 1, 512, 512, 256), device='cuda', dtype=torch.float32)
        buf177 = empty_strided((1, 256, 1, 1, 2), (512, 1, 512, 512, 256), device='cuda', dtype=torch.float32)
        buf178 = empty_strided((1, 256, 1, 1, 2), (512, 1, 512, 512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf173, buf174, buf175, buf176, buf177, buf178, 512, 98, grid=grid(512), stream=stream0)
        buf179 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf180 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf182 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf176, buf177, buf178, primals_528, primals_529, buf179, buf180, buf182, primals_528, primals_529, 256, 2, grid=grid(256), stream=stream0)
        del primals_528
        del primals_529
        # Source Nodes: [getattr_l__mod___layer1___0___downsample_0], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf114, primals_19, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf184 = reinterpret_tensor(buf171, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf171  # reuse
        # Source Nodes: [getattr_l__mod___layer1___0___downsample_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf183, buf184, 2048, 3136, grid=grid(2048, 3136), stream=stream0)
        buf185 = buf175; del buf175  # reuse
        buf186 = buf174; del buf174  # reuse
        buf187 = buf173; del buf173  # reuse
        # Source Nodes: [shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf184, buf185, buf186, buf187, 50176, 128, grid=grid(50176), stream=stream0)
        buf188 = buf178; del buf178  # reuse
        buf189 = buf177; del buf177  # reuse
        buf190 = buf176; del buf176  # reuse
        # Source Nodes: [shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf185, buf186, buf187, buf188, buf189, buf190, 512, 98, grid=grid(512), stream=stream0)
        buf191 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf192 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf194 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf188, buf189, buf190, primals_531, primals_532, buf191, buf192, buf194, primals_531, primals_532, 256, 2, grid=grid(256), stream=stream0)
        del primals_531
        del primals_532
        buf195 = reinterpret_tensor(buf183, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf183  # reuse
        buf196 = buf195; del buf195  # reuse
        # Source Nodes: [out_5, out_6, shortcut_1, shortcut_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_27.run(buf196, buf172, buf179, buf180, primals_17, primals_18, buf184, buf191, buf192, primals_20, primals_21, 6422528, grid=grid(6422528), stream=stream0)
        del primals_18
        del primals_21
        # Source Nodes: [out_8], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 104, 56, 56), (326144, 3136, 56, 1))
        buf198 = reinterpret_tensor(buf169, (8, 104, 56, 56), (326144, 1, 5824, 104), 0); del buf169  # reuse
        # Source Nodes: [out_8], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf197, buf198, 832, 3136, grid=grid(832, 3136), stream=stream0)
        buf199 = buf120; del buf120  # reuse
        buf200 = buf119; del buf119  # reuse
        buf201 = buf118; del buf118  # reuse
        # Source Nodes: [out_9], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf198, buf199, buf200, buf201, 20384, 128, grid=grid(20384), stream=stream0)
        buf202 = buf123; del buf123  # reuse
        buf203 = buf122; del buf122  # reuse
        buf204 = buf121; del buf121  # reuse
        # Source Nodes: [out_9], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf199, buf200, buf201, buf202, buf203, buf204, 208, 98, grid=grid(208), stream=stream0)
        buf205 = buf125; del buf125  # reuse
        buf206 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf208 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_9], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf202, buf203, buf204, primals_534, primals_535, buf205, buf206, buf208, primals_534, primals_535, 104, 2, grid=grid(104), stream=stream0)
        del primals_534
        del primals_535
        buf209 = reinterpret_tensor(buf197, (8, 104, 56, 56), (326144, 1, 5824, 104), 0); del buf197  # reuse
        buf2145 = empty_strided((8, 104, 56, 56), (326144, 1, 5824, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_10, out_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_16.run(buf198, buf205, buf206, primals_23, primals_24, buf209, buf2145, 2609152, grid=grid(2609152), stream=stream0)
        del primals_24
        # Source Nodes: [sp_14], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(reinterpret_tensor(buf209, (8, 26, 56, 56), (326144, 1, 5824, 104), 0), buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (8, 26, 56, 56), (81536, 3136, 56, 1))
        buf211 = reinterpret_tensor(buf155, (8, 26, 56, 56), (81536, 1, 1456, 26), 0); del buf155  # reuse
        # Source Nodes: [sp_14], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf210, buf211, 208, 3136, grid=grid(208, 3136), stream=stream0)
        buf212 = buf159; del buf159  # reuse
        buf213 = buf158; del buf158  # reuse
        buf214 = buf157; del buf157  # reuse
        # Source Nodes: [sp_15], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf211, buf212, buf213, buf214, 5096, 128, grid=grid(5096), stream=stream0)
        buf215 = buf162; del buf162  # reuse
        buf216 = buf161; del buf161  # reuse
        buf217 = buf160; del buf160  # reuse
        # Source Nodes: [sp_15], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf212, buf213, buf214, buf215, buf216, buf217, 52, 98, grid=grid(52), stream=stream0)
        buf218 = buf164; del buf164  # reuse
        buf219 = empty_strided((1, 26, 1, 1), (26, 1, 26, 26), device='cuda', dtype=torch.float32)
        buf221 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_15], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf215, buf216, buf217, primals_537, primals_538, buf218, buf219, buf221, primals_537, primals_538, 26, 2, grid=grid(26), stream=stream0)
        del primals_537
        del primals_538
        buf252 = empty((8, 104, 56, 56), device='cuda', dtype=torch.float32)
        buf222 = reinterpret_tensor(buf252, (8, 26, 56, 56), (326144, 3136, 56, 1), 0)  # alias
        buf2144 = empty_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_15, sp_16], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf211, buf218, buf219, primals_26, primals_27, buf222, buf2144, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del primals_27
        buf223 = reinterpret_tensor(buf210, (8, 26, 56, 56), (81536, 1, 1456, 26), 0); del buf210  # reuse
        # Source Nodes: [sp_17], Original ATen: [aten.add]
        triton_poi_fused_add_28.run(buf222, buf209, buf223, 25088, 26, grid=grid(25088, 26), stream=stream0)
        # Source Nodes: [sp_18], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (8, 26, 56, 56), (81536, 3136, 56, 1))
        buf225 = empty_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_18], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf224, buf225, 208, 3136, grid=grid(208, 3136), stream=stream0)
        buf226 = buf214; del buf214  # reuse
        buf227 = buf213; del buf213  # reuse
        buf228 = buf212; del buf212  # reuse
        # Source Nodes: [sp_19], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf225, buf226, buf227, buf228, 5096, 128, grid=grid(5096), stream=stream0)
        buf229 = buf217; del buf217  # reuse
        buf230 = buf216; del buf216  # reuse
        buf231 = buf215; del buf215  # reuse
        # Source Nodes: [sp_19], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf226, buf227, buf228, buf229, buf230, buf231, 52, 98, grid=grid(52), stream=stream0)
        buf232 = buf219; del buf219  # reuse
        buf233 = empty_strided((1, 26, 1, 1), (26, 1, 26, 26), device='cuda', dtype=torch.float32)
        buf235 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_19], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf229, buf230, buf231, primals_540, primals_541, buf232, buf233, buf235, primals_540, primals_541, 26, 2, grid=grid(26), stream=stream0)
        del primals_540
        del primals_541
        buf236 = reinterpret_tensor(buf252, (8, 26, 56, 56), (326144, 3136, 56, 1), 81536)  # alias
        buf2143 = empty_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_19, sp_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf225, buf232, buf233, primals_29, primals_30, buf236, buf2143, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del primals_30
        buf237 = reinterpret_tensor(buf224, (8, 26, 56, 56), (81536, 1, 1456, 26), 0); del buf224  # reuse
        # Source Nodes: [sp_21], Original ATen: [aten.add]
        triton_poi_fused_add_29.run(buf236, buf209, buf237, 25088, 26, grid=grid(25088, 26), stream=stream0)
        # Source Nodes: [sp_22], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (8, 26, 56, 56), (81536, 3136, 56, 1))
        buf239 = empty_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_22], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf238, buf239, 208, 3136, grid=grid(208, 3136), stream=stream0)
        buf240 = buf228; del buf228  # reuse
        buf241 = buf227; del buf227  # reuse
        buf242 = buf226; del buf226  # reuse
        # Source Nodes: [sp_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf239, buf240, buf241, buf242, 5096, 128, grid=grid(5096), stream=stream0)
        buf243 = buf231; del buf231  # reuse
        buf244 = buf230; del buf230  # reuse
        buf245 = buf229; del buf229  # reuse
        # Source Nodes: [sp_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf240, buf241, buf242, buf243, buf244, buf245, 52, 98, grid=grid(52), stream=stream0)
        buf246 = buf233; del buf233  # reuse
        buf247 = empty_strided((1, 26, 1, 1), (26, 1, 26, 26), device='cuda', dtype=torch.float32)
        buf249 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf243, buf244, buf245, primals_543, primals_544, buf246, buf247, buf249, primals_543, primals_544, 26, 2, grid=grid(26), stream=stream0)
        del primals_543
        del primals_544
        buf250 = reinterpret_tensor(buf252, (8, 26, 56, 56), (326144, 3136, 56, 1), 163072)  # alias
        buf2142 = empty_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_23, sp_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf239, buf246, buf247, primals_32, primals_33, buf250, buf2142, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del primals_33
        buf251 = reinterpret_tensor(buf252, (8, 26, 56, 56), (326144, 3136, 56, 1), 244608)  # alias
        # Source Nodes: [cat_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_30.run(buf209, buf251, 208, 3136, grid=grid(208, 3136), stream=stream0)
        buf253 = empty_strided((8, 104, 56, 56), (326144, 1, 5824, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_64], Original ATen: [aten.cat]
        triton_poi_fused_convolution_12.run(buf252, buf253, 832, 3136, grid=grid(832, 3136), stream=stream0)
        del buf222
        del buf236
        del buf250
        del buf251
        # Source Nodes: [out_12], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf255 = empty_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_12], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf254, buf255, 2048, 3136, grid=grid(2048, 3136), stream=stream0)
        buf256 = buf187; del buf187  # reuse
        buf257 = buf186; del buf186  # reuse
        buf258 = buf185; del buf185  # reuse
        # Source Nodes: [out_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf255, buf256, buf257, buf258, 50176, 128, grid=grid(50176), stream=stream0)
        buf259 = buf190; del buf190  # reuse
        buf260 = buf189; del buf189  # reuse
        buf261 = buf188; del buf188  # reuse
        # Source Nodes: [out_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf256, buf257, buf258, buf259, buf260, buf261, 512, 98, grid=grid(512), stream=stream0)
        buf262 = buf192; del buf192  # reuse
        buf263 = buf180; del buf180  # reuse
        buf265 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf259, buf260, buf261, primals_546, primals_547, buf262, buf263, buf265, primals_546, primals_547, 256, 2, grid=grid(256), stream=stream0)
        del primals_546
        del primals_547
        buf266 = reinterpret_tensor(buf254, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf254  # reuse
        # Source Nodes: [out_13, out_14, shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_31.run(buf255, buf262, buf263, primals_35, primals_36, buf196, buf266, 6422528, grid=grid(6422528), stream=stream0)
        del primals_36
        # Source Nodes: [out_16], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf266, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (8, 104, 56, 56), (326144, 3136, 56, 1))
        buf268 = reinterpret_tensor(buf252, (8, 104, 56, 56), (326144, 1, 5824, 104), 0); del buf252  # reuse
        # Source Nodes: [out_16], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf267, buf268, 832, 3136, grid=grid(832, 3136), stream=stream0)
        buf269 = buf201; del buf201  # reuse
        buf270 = buf200; del buf200  # reuse
        buf271 = buf199; del buf199  # reuse
        # Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf268, buf269, buf270, buf271, 20384, 128, grid=grid(20384), stream=stream0)
        buf272 = buf204; del buf204  # reuse
        buf273 = buf203; del buf203  # reuse
        buf274 = buf202; del buf202  # reuse
        # Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf269, buf270, buf271, buf272, buf273, buf274, 208, 98, grid=grid(208), stream=stream0)
        buf275 = buf206; del buf206  # reuse
        buf276 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf278 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf272, buf273, buf274, primals_549, primals_550, buf275, buf276, buf278, primals_549, primals_550, 104, 2, grid=grid(104), stream=stream0)
        del primals_549
        del primals_550
        buf279 = reinterpret_tensor(buf267, (8, 104, 56, 56), (326144, 1, 5824, 104), 0); del buf267  # reuse
        buf2141 = empty_strided((8, 104, 56, 56), (326144, 1, 5824, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_17, out_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_16.run(buf268, buf275, buf276, primals_38, primals_39, buf279, buf2141, 2609152, grid=grid(2609152), stream=stream0)
        del primals_39
        # Source Nodes: [sp_27], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(reinterpret_tensor(buf279, (8, 26, 56, 56), (326144, 1, 5824, 104), 0), buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (8, 26, 56, 56), (81536, 3136, 56, 1))
        buf281 = reinterpret_tensor(buf238, (8, 26, 56, 56), (81536, 1, 1456, 26), 0); del buf238  # reuse
        # Source Nodes: [sp_27], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf280, buf281, 208, 3136, grid=grid(208, 3136), stream=stream0)
        buf282 = buf242; del buf242  # reuse
        buf283 = buf241; del buf241  # reuse
        buf284 = buf240; del buf240  # reuse
        # Source Nodes: [sp_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf281, buf282, buf283, buf284, 5096, 128, grid=grid(5096), stream=stream0)
        buf285 = buf245; del buf245  # reuse
        buf286 = buf244; del buf244  # reuse
        buf287 = buf243; del buf243  # reuse
        # Source Nodes: [sp_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf282, buf283, buf284, buf285, buf286, buf287, 52, 98, grid=grid(52), stream=stream0)
        buf288 = buf247; del buf247  # reuse
        buf289 = empty_strided((1, 26, 1, 1), (26, 1, 26, 26), device='cuda', dtype=torch.float32)
        buf291 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf285, buf286, buf287, primals_552, primals_553, buf288, buf289, buf291, primals_552, primals_553, 26, 2, grid=grid(26), stream=stream0)
        del primals_552
        del primals_553
        buf322 = empty((8, 104, 56, 56), device='cuda', dtype=torch.float32)
        buf292 = reinterpret_tensor(buf322, (8, 26, 56, 56), (326144, 3136, 56, 1), 0)  # alias
        buf2140 = empty_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_28, sp_29], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf281, buf288, buf289, primals_41, primals_42, buf292, buf2140, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del primals_42
        buf293 = reinterpret_tensor(buf280, (8, 26, 56, 56), (81536, 1, 1456, 26), 0); del buf280  # reuse
        # Source Nodes: [sp_30], Original ATen: [aten.add]
        triton_poi_fused_add_28.run(buf292, buf279, buf293, 25088, 26, grid=grid(25088, 26), stream=stream0)
        # Source Nodes: [sp_31], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (8, 26, 56, 56), (81536, 3136, 56, 1))
        buf295 = empty_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_31], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf294, buf295, 208, 3136, grid=grid(208, 3136), stream=stream0)
        buf296 = buf284; del buf284  # reuse
        buf297 = buf283; del buf283  # reuse
        buf298 = buf282; del buf282  # reuse
        # Source Nodes: [sp_32], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf295, buf296, buf297, buf298, 5096, 128, grid=grid(5096), stream=stream0)
        buf299 = buf287; del buf287  # reuse
        buf300 = buf286; del buf286  # reuse
        buf301 = buf285; del buf285  # reuse
        # Source Nodes: [sp_32], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf296, buf297, buf298, buf299, buf300, buf301, 52, 98, grid=grid(52), stream=stream0)
        buf302 = buf289; del buf289  # reuse
        buf303 = empty_strided((1, 26, 1, 1), (26, 1, 26, 26), device='cuda', dtype=torch.float32)
        buf305 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_32], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf299, buf300, buf301, primals_555, primals_556, buf302, buf303, buf305, primals_555, primals_556, 26, 2, grid=grid(26), stream=stream0)
        del primals_555
        del primals_556
        buf306 = reinterpret_tensor(buf322, (8, 26, 56, 56), (326144, 3136, 56, 1), 81536)  # alias
        buf2139 = empty_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_32, sp_33], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf295, buf302, buf303, primals_44, primals_45, buf306, buf2139, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del primals_45
        buf307 = reinterpret_tensor(buf294, (8, 26, 56, 56), (81536, 1, 1456, 26), 0); del buf294  # reuse
        # Source Nodes: [sp_34], Original ATen: [aten.add]
        triton_poi_fused_add_29.run(buf306, buf279, buf307, 25088, 26, grid=grid(25088, 26), stream=stream0)
        # Source Nodes: [sp_35], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (8, 26, 56, 56), (81536, 3136, 56, 1))
        buf309 = empty_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_35], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf308, buf309, 208, 3136, grid=grid(208, 3136), stream=stream0)
        buf310 = buf298; del buf298  # reuse
        buf311 = buf297; del buf297  # reuse
        buf312 = buf296; del buf296  # reuse
        # Source Nodes: [sp_36], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf309, buf310, buf311, buf312, 5096, 128, grid=grid(5096), stream=stream0)
        buf313 = buf301; del buf301  # reuse
        buf314 = buf300; del buf300  # reuse
        buf315 = buf299; del buf299  # reuse
        # Source Nodes: [sp_36], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf310, buf311, buf312, buf313, buf314, buf315, 52, 98, grid=grid(52), stream=stream0)
        del buf310
        del buf311
        del buf312
        buf316 = buf303; del buf303  # reuse
        buf317 = empty_strided((1, 26, 1, 1), (26, 1, 26, 26), device='cuda', dtype=torch.float32)
        buf319 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_36], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf313, buf314, buf315, primals_558, primals_559, buf316, buf317, buf319, primals_558, primals_559, 26, 2, grid=grid(26), stream=stream0)
        del primals_558
        del primals_559
        buf320 = reinterpret_tensor(buf322, (8, 26, 56, 56), (326144, 3136, 56, 1), 163072)  # alias
        buf2138 = empty_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_36, sp_37], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf309, buf316, buf317, primals_47, primals_48, buf320, buf2138, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del buf317
        del primals_48
        buf321 = reinterpret_tensor(buf322, (8, 26, 56, 56), (326144, 3136, 56, 1), 244608)  # alias
        # Source Nodes: [cat_63], Original ATen: [aten.cat]
        triton_poi_fused_cat_30.run(buf279, buf321, 208, 3136, grid=grid(208, 3136), stream=stream0)
        buf323 = empty_strided((8, 104, 56, 56), (326144, 1, 5824, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_63], Original ATen: [aten.cat]
        triton_poi_fused_convolution_12.run(buf322, buf323, 832, 3136, grid=grid(832, 3136), stream=stream0)
        del buf292
        del buf306
        del buf320
        del buf321
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        buf324 = extern_kernels.convolution(buf323, primals_49, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf325 = empty_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf324, buf325, 2048, 3136, grid=grid(2048, 3136), stream=stream0)
        buf326 = buf258; del buf258  # reuse
        buf327 = buf257; del buf257  # reuse
        buf328 = buf256; del buf256  # reuse
        # Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf325, buf326, buf327, buf328, 50176, 128, grid=grid(50176), stream=stream0)
        buf329 = buf261; del buf261  # reuse
        buf330 = buf260; del buf260  # reuse
        buf331 = buf259; del buf259  # reuse
        # Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf326, buf327, buf328, buf329, buf330, buf331, 512, 98, grid=grid(512), stream=stream0)
        del buf326
        del buf327
        del buf328
        buf332 = buf263; del buf263  # reuse
        buf333 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf335 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf329, buf330, buf331, primals_561, primals_562, buf332, buf333, buf335, primals_561, primals_562, 256, 2, grid=grid(256), stream=stream0)
        del primals_561
        del primals_562
        buf336 = reinterpret_tensor(buf324, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf324  # reuse
        # Source Nodes: [out_21, out_22, shortcut_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_31.run(buf325, buf332, buf333, primals_50, primals_51, buf266, buf336, 6422528, grid=grid(6422528), stream=stream0)
        del buf333
        del primals_51
        # Source Nodes: [out_24], Original ATen: [aten.convolution]
        buf337 = extern_kernels.convolution(buf336, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf337, (8, 208, 56, 56), (652288, 3136, 56, 1))
        buf338 = empty_strided((8, 208, 56, 56), (652288, 1, 11648, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_24], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_32.run(buf337, buf338, 1664, 3136, grid=grid(1664, 3136), stream=stream0)
        buf339 = empty_strided((1, 208, 1, 1, 196), (40768, 1, 40768, 40768, 208), device='cuda', dtype=torch.float32)
        buf340 = empty_strided((1, 208, 1, 1, 196), (40768, 1, 40768, 40768, 208), device='cuda', dtype=torch.float32)
        buf341 = empty_strided((1, 208, 1, 1, 196), (40768, 1, 40768, 40768, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf338, buf339, buf340, buf341, 40768, 128, grid=grid(40768), stream=stream0)
        buf342 = empty_strided((1, 208, 1, 1, 2), (416, 1, 416, 416, 208), device='cuda', dtype=torch.float32)
        buf343 = empty_strided((1, 208, 1, 1, 2), (416, 1, 416, 416, 208), device='cuda', dtype=torch.float32)
        buf344 = empty_strided((1, 208, 1, 1, 2), (416, 1, 416, 416, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_34.run(buf339, buf340, buf341, buf342, buf343, buf344, 416, 98, grid=grid(416), stream=stream0)
        del buf339
        del buf340
        del buf341
        buf345 = reinterpret_tensor(buf274, (1, 208, 1, 1), (208, 1, 208, 208), 0); del buf274  # reuse
        buf346 = reinterpret_tensor(buf273, (1, 208, 1, 1), (208, 1, 208, 208), 0); del buf273  # reuse
        buf348 = reinterpret_tensor(buf272, (208, ), (1, ), 0); del buf272  # reuse
        # Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf342, buf343, buf344, primals_564, primals_565, buf345, buf346, buf348, primals_564, primals_565, 208, 2, grid=grid(208), stream=stream0)
        del primals_564
        del primals_565
        buf349 = reinterpret_tensor(buf337, (8, 208, 56, 56), (652288, 1, 11648, 208), 0); del buf337  # reuse
        buf2137 = empty_strided((8, 208, 56, 56), (652288, 1, 11648, 208), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_25, out_26], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_36.run(buf338, buf345, buf346, primals_53, primals_54, buf349, buf2137, 5218304, grid=grid(5218304), stream=stream0)
        del primals_54
        # Source Nodes: [sp_40], Original ATen: [aten.convolution]
        buf350 = extern_kernels.convolution(reinterpret_tensor(buf349, (8, 52, 56, 56), (652288, 1, 11648, 208), 0), buf10, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf351 = empty_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_40], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf350, buf351, 416, 784, grid=grid(416, 784), stream=stream0)
        buf352 = empty_strided((1, 52, 1, 1, 49), (2548, 1, 2548, 2548, 52), device='cuda', dtype=torch.float32)
        buf353 = empty_strided((1, 52, 1, 1, 49), (2548, 1, 2548, 2548, 52), device='cuda', dtype=torch.float32)
        buf354 = empty_strided((1, 52, 1, 1, 49), (2548, 1, 2548, 2548, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf351, buf352, buf353, buf354, 2548, 128, grid=grid(2548), stream=stream0)
        buf355 = reinterpret_tensor(buf315, (1, 52, 1, 1), (52, 1, 52, 52), 0); del buf315  # reuse
        buf356 = reinterpret_tensor(buf314, (1, 52, 1, 1), (52, 1, 52, 52), 0); del buf314  # reuse
        buf358 = reinterpret_tensor(buf313, (52, ), (1, ), 0); del buf313  # reuse
        # Source Nodes: [sp_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf352, buf353, buf354, primals_567, primals_568, buf355, buf356, buf358, primals_567, primals_568, 52, 49, grid=grid(52), stream=stream0)
        del primals_567
        del primals_568
        buf381 = empty((8, 208, 28, 28), device='cuda', dtype=torch.float32)
        buf359 = reinterpret_tensor(buf381, (8, 52, 28, 28), (163072, 784, 28, 1), 0)  # alias
        buf2136 = empty_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_41, sp_42], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_40.run(buf351, buf355, buf356, primals_56, primals_57, buf359, buf2136, 416, 784, grid=grid(416, 784), stream=stream0)
        del primals_57
        # Source Nodes: [sp_44], Original ATen: [aten.convolution]
        buf360 = extern_kernels.convolution(reinterpret_tensor(buf349, (8, 52, 56, 56), (652288, 1, 11648, 208), 52), buf11, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf360, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf361 = reinterpret_tensor(buf350, (8, 52, 28, 28), (40768, 1, 1456, 52), 0); del buf350  # reuse
        # Source Nodes: [sp_44], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf360, buf361, 416, 784, grid=grid(416, 784), stream=stream0)
        buf362 = buf354; del buf354  # reuse
        buf363 = buf353; del buf353  # reuse
        buf364 = buf352; del buf352  # reuse
        # Source Nodes: [sp_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf361, buf362, buf363, buf364, 2548, 128, grid=grid(2548), stream=stream0)
        buf365 = buf356; del buf356  # reuse
        buf366 = empty_strided((1, 52, 1, 1), (52, 1, 52, 52), device='cuda', dtype=torch.float32)
        buf368 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf362, buf363, buf364, primals_570, primals_571, buf365, buf366, buf368, primals_570, primals_571, 52, 49, grid=grid(52), stream=stream0)
        del primals_570
        del primals_571
        buf369 = reinterpret_tensor(buf381, (8, 52, 28, 28), (163072, 784, 28, 1), 40768)  # alias
        buf2135 = empty_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_45, sp_46], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_40.run(buf361, buf365, buf366, primals_59, primals_60, buf369, buf2135, 416, 784, grid=grid(416, 784), stream=stream0)
        del primals_60
        # Source Nodes: [sp_48], Original ATen: [aten.convolution]
        buf370 = extern_kernels.convolution(reinterpret_tensor(buf349, (8, 52, 56, 56), (652288, 1, 11648, 208), 104), buf12, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf370, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf371 = reinterpret_tensor(buf360, (8, 52, 28, 28), (40768, 1, 1456, 52), 0); del buf360  # reuse
        # Source Nodes: [sp_48], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf370, buf371, 416, 784, grid=grid(416, 784), stream=stream0)
        buf372 = buf364; del buf364  # reuse
        buf373 = buf363; del buf363  # reuse
        buf374 = buf362; del buf362  # reuse
        # Source Nodes: [sp_49], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf371, buf372, buf373, buf374, 2548, 128, grid=grid(2548), stream=stream0)
        buf375 = buf366; del buf366  # reuse
        buf376 = empty_strided((1, 52, 1, 1), (52, 1, 52, 52), device='cuda', dtype=torch.float32)
        buf378 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_49], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf372, buf373, buf374, primals_573, primals_574, buf375, buf376, buf378, primals_573, primals_574, 52, 49, grid=grid(52), stream=stream0)
        del primals_573
        del primals_574
        buf379 = reinterpret_tensor(buf381, (8, 52, 28, 28), (163072, 784, 28, 1), 81536)  # alias
        buf2134 = empty_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_49, sp_50], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_40.run(buf371, buf375, buf376, primals_62, primals_63, buf379, buf2134, 416, 784, grid=grid(416, 784), stream=stream0)
        del primals_63
        buf380 = reinterpret_tensor(buf381, (8, 52, 28, 28), (163072, 784, 28, 1), 122304)  # alias
        # Source Nodes: [getattr_l__mod___layer2___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_41.run(buf349, buf380, 416, 784, grid=grid(416, 784), stream=stream0)
        buf382 = empty_strided((8, 208, 28, 28), (163072, 1, 5824, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_62], Original ATen: [aten.cat]
        triton_poi_fused_cat_42.run(buf381, buf382, 1664, 784, grid=grid(1664, 784), stream=stream0)
        del buf359
        del buf369
        del buf379
        del buf380
        # Source Nodes: [out_28], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf383, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf384 = empty_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_28], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf383, buf384, 4096, 784, grid=grid(4096, 784), stream=stream0)
        buf385 = empty_strided((1, 512, 1, 1, 49), (25088, 1, 25088, 25088, 512), device='cuda', dtype=torch.float32)
        buf386 = empty_strided((1, 512, 1, 1, 49), (25088, 1, 25088, 25088, 512), device='cuda', dtype=torch.float32)
        buf387 = empty_strided((1, 512, 1, 1, 49), (25088, 1, 25088, 25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf384, buf385, buf386, buf387, 25088, 128, grid=grid(25088), stream=stream0)
        buf388 = reinterpret_tensor(buf331, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf331  # reuse
        buf389 = reinterpret_tensor(buf330, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf330  # reuse
        buf391 = reinterpret_tensor(buf329, (512, ), (1, ), 0); del buf329  # reuse
        # Source Nodes: [out_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf385, buf386, buf387, primals_576, primals_577, buf388, buf389, buf391, primals_576, primals_577, 512, 49, grid=grid(512), stream=stream0)
        del primals_576
        del primals_577
        # Source Nodes: [getattr_l__mod___layer2___0___downsample_0], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf336, primals_67, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf393 = reinterpret_tensor(buf383, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf383  # reuse
        # Source Nodes: [getattr_l__mod___layer2___0___downsample_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf392, buf393, 4096, 784, grid=grid(4096, 784), stream=stream0)
        buf394 = buf387; del buf387  # reuse
        buf395 = buf386; del buf386  # reuse
        buf396 = buf385; del buf385  # reuse
        # Source Nodes: [shortcut_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf393, buf394, buf395, buf396, 25088, 128, grid=grid(25088), stream=stream0)
        buf397 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf398 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf400 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf394, buf395, buf396, primals_579, primals_580, buf397, buf398, buf400, primals_579, primals_580, 512, 49, grid=grid(512), stream=stream0)
        del primals_579
        del primals_580
        buf401 = reinterpret_tensor(buf392, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf392  # reuse
        buf402 = buf401; del buf401  # reuse
        # Source Nodes: [out_29, out_30, shortcut_5, shortcut_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_46.run(buf402, buf384, buf388, buf389, primals_65, primals_66, buf393, buf397, buf398, primals_68, primals_69, 3211264, grid=grid(3211264), stream=stream0)
        del primals_66
        del primals_69
        # Source Nodes: [out_32], Original ATen: [aten.convolution]
        buf403 = extern_kernels.convolution(buf402, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf403, (8, 208, 28, 28), (163072, 784, 28, 1))
        buf404 = reinterpret_tensor(buf381, (8, 208, 28, 28), (163072, 1, 5824, 208), 0); del buf381  # reuse
        # Source Nodes: [out_32], Original ATen: [aten.convolution]
        triton_poi_fused_cat_42.run(buf403, buf404, 1664, 784, grid=grid(1664, 784), stream=stream0)
        buf405 = empty_strided((1, 208, 1, 1, 49), (10192, 1, 10192, 10192, 208), device='cuda', dtype=torch.float32)
        buf406 = empty_strided((1, 208, 1, 1, 49), (10192, 1, 10192, 10192, 208), device='cuda', dtype=torch.float32)
        buf407 = empty_strided((1, 208, 1, 1, 49), (10192, 1, 10192, 10192, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_47.run(buf404, buf405, buf406, buf407, 10192, 128, grid=grid(10192), stream=stream0)
        buf408 = buf346; del buf346  # reuse
        buf409 = empty_strided((1, 208, 1, 1), (208, 1, 208, 208), device='cuda', dtype=torch.float32)
        buf411 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_48.run(buf405, buf406, buf407, primals_582, primals_583, buf408, buf409, buf411, primals_582, primals_583, 208, 49, grid=grid(208), stream=stream0)
        del primals_582
        del primals_583
        buf412 = reinterpret_tensor(buf403, (8, 208, 28, 28), (163072, 1, 5824, 208), 0); del buf403  # reuse
        buf2133 = empty_strided((8, 208, 28, 28), (163072, 1, 5824, 208), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_33, out_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_49.run(buf404, buf408, buf409, primals_71, primals_72, buf412, buf2133, 1304576, grid=grid(1304576), stream=stream0)
        del primals_72
        # Source Nodes: [sp_53], Original ATen: [aten.convolution]
        buf413 = extern_kernels.convolution(reinterpret_tensor(buf412, (8, 52, 28, 28), (163072, 1, 5824, 208), 0), buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf413, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf414 = reinterpret_tensor(buf370, (8, 52, 28, 28), (40768, 1, 1456, 52), 0); del buf370  # reuse
        # Source Nodes: [sp_53], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf413, buf414, 416, 784, grid=grid(416, 784), stream=stream0)
        buf415 = buf374; del buf374  # reuse
        buf416 = buf373; del buf373  # reuse
        buf417 = buf372; del buf372  # reuse
        # Source Nodes: [sp_54], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf414, buf415, buf416, buf417, 2548, 128, grid=grid(2548), stream=stream0)
        buf418 = buf376; del buf376  # reuse
        buf419 = empty_strided((1, 52, 1, 1), (52, 1, 52, 52), device='cuda', dtype=torch.float32)
        buf421 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_54], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf415, buf416, buf417, primals_585, primals_586, buf418, buf419, buf421, primals_585, primals_586, 52, 49, grid=grid(52), stream=stream0)
        del primals_585
        del primals_586
        buf446 = empty((8, 208, 28, 28), device='cuda', dtype=torch.float32)
        buf422 = reinterpret_tensor(buf446, (8, 52, 28, 28), (163072, 784, 28, 1), 0)  # alias
        buf2132 = empty_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_54, sp_55], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_40.run(buf414, buf418, buf419, primals_74, primals_75, buf422, buf2132, 416, 784, grid=grid(416, 784), stream=stream0)
        del primals_75
        buf423 = reinterpret_tensor(buf413, (8, 52, 28, 28), (40768, 1, 1456, 52), 0); del buf413  # reuse
        # Source Nodes: [sp_56], Original ATen: [aten.add]
        triton_poi_fused_add_50.run(buf422, buf412, buf423, 6272, 52, grid=grid(6272, 52), stream=stream0)
        # Source Nodes: [sp_57], Original ATen: [aten.convolution]
        buf424 = extern_kernels.convolution(buf423, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf424, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf425 = empty_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_57], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf424, buf425, 416, 784, grid=grid(416, 784), stream=stream0)
        buf426 = buf417; del buf417  # reuse
        buf427 = buf416; del buf416  # reuse
        buf428 = buf415; del buf415  # reuse
        # Source Nodes: [sp_58], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf425, buf426, buf427, buf428, 2548, 128, grid=grid(2548), stream=stream0)
        buf429 = buf419; del buf419  # reuse
        buf430 = empty_strided((1, 52, 1, 1), (52, 1, 52, 52), device='cuda', dtype=torch.float32)
        buf432 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_58], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf426, buf427, buf428, primals_588, primals_589, buf429, buf430, buf432, primals_588, primals_589, 52, 49, grid=grid(52), stream=stream0)
        del primals_588
        del primals_589
        buf433 = reinterpret_tensor(buf446, (8, 52, 28, 28), (163072, 784, 28, 1), 40768)  # alias
        buf2131 = empty_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_58, sp_59], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_40.run(buf425, buf429, buf430, primals_77, primals_78, buf433, buf2131, 416, 784, grid=grid(416, 784), stream=stream0)
        del primals_78
        buf434 = reinterpret_tensor(buf424, (8, 52, 28, 28), (40768, 1, 1456, 52), 0); del buf424  # reuse
        # Source Nodes: [sp_60], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(buf433, buf412, buf434, 6272, 52, grid=grid(6272, 52), stream=stream0)
        # Source Nodes: [sp_61], Original ATen: [aten.convolution]
        buf435 = extern_kernels.convolution(buf434, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf435, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf436 = empty_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_61], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf435, buf436, 416, 784, grid=grid(416, 784), stream=stream0)
        buf437 = buf428; del buf428  # reuse
        buf438 = buf427; del buf427  # reuse
        buf439 = buf426; del buf426  # reuse
        # Source Nodes: [sp_62], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf436, buf437, buf438, buf439, 2548, 128, grid=grid(2548), stream=stream0)
        buf440 = buf430; del buf430  # reuse
        buf441 = empty_strided((1, 52, 1, 1), (52, 1, 52, 52), device='cuda', dtype=torch.float32)
        buf443 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_62], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf437, buf438, buf439, primals_591, primals_592, buf440, buf441, buf443, primals_591, primals_592, 52, 49, grid=grid(52), stream=stream0)
        del primals_591
        del primals_592
        buf444 = reinterpret_tensor(buf446, (8, 52, 28, 28), (163072, 784, 28, 1), 81536)  # alias
        buf2130 = empty_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_62, sp_63], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_40.run(buf436, buf440, buf441, primals_80, primals_81, buf444, buf2130, 416, 784, grid=grid(416, 784), stream=stream0)
        del primals_81
        buf445 = reinterpret_tensor(buf446, (8, 52, 28, 28), (163072, 784, 28, 1), 122304)  # alias
        # Source Nodes: [cat_61], Original ATen: [aten.cat]
        triton_poi_fused_cat_52.run(buf412, buf445, 416, 784, grid=grid(416, 784), stream=stream0)
        buf447 = empty_strided((8, 208, 28, 28), (163072, 1, 5824, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_61], Original ATen: [aten.cat]
        triton_poi_fused_cat_42.run(buf446, buf447, 1664, 784, grid=grid(1664, 784), stream=stream0)
        del buf422
        del buf433
        del buf444
        del buf445
        # Source Nodes: [out_36], Original ATen: [aten.convolution]
        buf448 = extern_kernels.convolution(buf447, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf448, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf449 = empty_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_36], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf448, buf449, 4096, 784, grid=grid(4096, 784), stream=stream0)
        buf450 = buf396; del buf396  # reuse
        buf451 = buf395; del buf395  # reuse
        buf452 = buf394; del buf394  # reuse
        # Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf449, buf450, buf451, buf452, 25088, 128, grid=grid(25088), stream=stream0)
        buf453 = buf398; del buf398  # reuse
        buf454 = buf389; del buf389  # reuse
        buf456 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf450, buf451, buf452, primals_594, primals_595, buf453, buf454, buf456, primals_594, primals_595, 512, 49, grid=grid(512), stream=stream0)
        del primals_594
        del primals_595
        buf457 = reinterpret_tensor(buf448, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf448  # reuse
        # Source Nodes: [out_37, out_38, shortcut_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_53.run(buf449, buf453, buf454, primals_83, primals_84, buf402, buf457, 3211264, grid=grid(3211264), stream=stream0)
        del primals_84
        # Source Nodes: [out_40], Original ATen: [aten.convolution]
        buf458 = extern_kernels.convolution(buf457, primals_85, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf458, (8, 208, 28, 28), (163072, 784, 28, 1))
        buf459 = reinterpret_tensor(buf446, (8, 208, 28, 28), (163072, 1, 5824, 208), 0); del buf446  # reuse
        # Source Nodes: [out_40], Original ATen: [aten.convolution]
        triton_poi_fused_cat_42.run(buf458, buf459, 1664, 784, grid=grid(1664, 784), stream=stream0)
        buf460 = buf407; del buf407  # reuse
        buf461 = buf406; del buf406  # reuse
        buf462 = buf405; del buf405  # reuse
        # Source Nodes: [out_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_47.run(buf459, buf460, buf461, buf462, 10192, 128, grid=grid(10192), stream=stream0)
        buf463 = buf409; del buf409  # reuse
        buf464 = empty_strided((1, 208, 1, 1), (208, 1, 208, 208), device='cuda', dtype=torch.float32)
        buf466 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_48.run(buf460, buf461, buf462, primals_597, primals_598, buf463, buf464, buf466, primals_597, primals_598, 208, 49, grid=grid(208), stream=stream0)
        del primals_597
        del primals_598
        buf467 = reinterpret_tensor(buf458, (8, 208, 28, 28), (163072, 1, 5824, 208), 0); del buf458  # reuse
        buf2129 = empty_strided((8, 208, 28, 28), (163072, 1, 5824, 208), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_41, out_42], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_49.run(buf459, buf463, buf464, primals_86, primals_87, buf467, buf2129, 1304576, grid=grid(1304576), stream=stream0)
        del primals_87
        # Source Nodes: [sp_66], Original ATen: [aten.convolution]
        buf468 = extern_kernels.convolution(reinterpret_tensor(buf467, (8, 52, 28, 28), (163072, 1, 5824, 208), 0), buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf468, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf469 = reinterpret_tensor(buf435, (8, 52, 28, 28), (40768, 1, 1456, 52), 0); del buf435  # reuse
        # Source Nodes: [sp_66], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf468, buf469, 416, 784, grid=grid(416, 784), stream=stream0)
        buf470 = buf439; del buf439  # reuse
        buf471 = buf438; del buf438  # reuse
        buf472 = buf437; del buf437  # reuse
        # Source Nodes: [sp_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf469, buf470, buf471, buf472, 2548, 128, grid=grid(2548), stream=stream0)
        buf473 = buf441; del buf441  # reuse
        buf474 = empty_strided((1, 52, 1, 1), (52, 1, 52, 52), device='cuda', dtype=torch.float32)
        buf476 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf470, buf471, buf472, primals_600, primals_601, buf473, buf474, buf476, primals_600, primals_601, 52, 49, grid=grid(52), stream=stream0)
        del primals_600
        del primals_601
        buf501 = empty((8, 208, 28, 28), device='cuda', dtype=torch.float32)
        buf477 = reinterpret_tensor(buf501, (8, 52, 28, 28), (163072, 784, 28, 1), 0)  # alias
        buf2128 = empty_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_67, sp_68], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_40.run(buf469, buf473, buf474, primals_89, primals_90, buf477, buf2128, 416, 784, grid=grid(416, 784), stream=stream0)
        del primals_90
        buf478 = reinterpret_tensor(buf468, (8, 52, 28, 28), (40768, 1, 1456, 52), 0); del buf468  # reuse
        # Source Nodes: [sp_69], Original ATen: [aten.add]
        triton_poi_fused_add_50.run(buf477, buf467, buf478, 6272, 52, grid=grid(6272, 52), stream=stream0)
        # Source Nodes: [sp_70], Original ATen: [aten.convolution]
        buf479 = extern_kernels.convolution(buf478, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf479, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf480 = empty_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_70], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf479, buf480, 416, 784, grid=grid(416, 784), stream=stream0)
        buf481 = buf472; del buf472  # reuse
        buf482 = buf471; del buf471  # reuse
        buf483 = buf470; del buf470  # reuse
        # Source Nodes: [sp_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf480, buf481, buf482, buf483, 2548, 128, grid=grid(2548), stream=stream0)
        buf484 = buf474; del buf474  # reuse
        buf485 = empty_strided((1, 52, 1, 1), (52, 1, 52, 52), device='cuda', dtype=torch.float32)
        buf487 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf481, buf482, buf483, primals_603, primals_604, buf484, buf485, buf487, primals_603, primals_604, 52, 49, grid=grid(52), stream=stream0)
        del primals_603
        del primals_604
        buf488 = reinterpret_tensor(buf501, (8, 52, 28, 28), (163072, 784, 28, 1), 40768)  # alias
        buf2127 = empty_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_71, sp_72], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_40.run(buf480, buf484, buf485, primals_92, primals_93, buf488, buf2127, 416, 784, grid=grid(416, 784), stream=stream0)
        del primals_93
        buf489 = reinterpret_tensor(buf479, (8, 52, 28, 28), (40768, 1, 1456, 52), 0); del buf479  # reuse
        # Source Nodes: [sp_73], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(buf488, buf467, buf489, 6272, 52, grid=grid(6272, 52), stream=stream0)
        # Source Nodes: [sp_74], Original ATen: [aten.convolution]
        buf490 = extern_kernels.convolution(buf489, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf490, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf491 = empty_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_74], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf490, buf491, 416, 784, grid=grid(416, 784), stream=stream0)
        buf492 = buf483; del buf483  # reuse
        buf493 = buf482; del buf482  # reuse
        buf494 = buf481; del buf481  # reuse
        # Source Nodes: [sp_75], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf491, buf492, buf493, buf494, 2548, 128, grid=grid(2548), stream=stream0)
        buf495 = buf485; del buf485  # reuse
        buf496 = empty_strided((1, 52, 1, 1), (52, 1, 52, 52), device='cuda', dtype=torch.float32)
        buf498 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_75], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf492, buf493, buf494, primals_606, primals_607, buf495, buf496, buf498, primals_606, primals_607, 52, 49, grid=grid(52), stream=stream0)
        del primals_606
        del primals_607
        buf499 = reinterpret_tensor(buf501, (8, 52, 28, 28), (163072, 784, 28, 1), 81536)  # alias
        buf2126 = empty_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_75, sp_76], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_40.run(buf491, buf495, buf496, primals_95, primals_96, buf499, buf2126, 416, 784, grid=grid(416, 784), stream=stream0)
        del primals_96
        buf500 = reinterpret_tensor(buf501, (8, 52, 28, 28), (163072, 784, 28, 1), 122304)  # alias
        # Source Nodes: [cat_60], Original ATen: [aten.cat]
        triton_poi_fused_cat_52.run(buf467, buf500, 416, 784, grid=grid(416, 784), stream=stream0)
        buf502 = empty_strided((8, 208, 28, 28), (163072, 1, 5824, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_60], Original ATen: [aten.cat]
        triton_poi_fused_cat_42.run(buf501, buf502, 1664, 784, grid=grid(1664, 784), stream=stream0)
        del buf477
        del buf488
        del buf499
        del buf500
        # Source Nodes: [out_44], Original ATen: [aten.convolution]
        buf503 = extern_kernels.convolution(buf502, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf503, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf504 = empty_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_44], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf503, buf504, 4096, 784, grid=grid(4096, 784), stream=stream0)
        buf505 = buf452; del buf452  # reuse
        buf506 = buf451; del buf451  # reuse
        buf507 = buf450; del buf450  # reuse
        # Source Nodes: [out_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf504, buf505, buf506, buf507, 25088, 128, grid=grid(25088), stream=stream0)
        buf508 = buf454; del buf454  # reuse
        buf509 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf511 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf505, buf506, buf507, primals_609, primals_610, buf508, buf509, buf511, primals_609, primals_610, 512, 49, grid=grid(512), stream=stream0)
        del primals_609
        del primals_610
        buf512 = reinterpret_tensor(buf503, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf503  # reuse
        # Source Nodes: [out_45, out_46, shortcut_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_53.run(buf504, buf508, buf509, primals_98, primals_99, buf457, buf512, 3211264, grid=grid(3211264), stream=stream0)
        del primals_99
        # Source Nodes: [out_48], Original ATen: [aten.convolution]
        buf513 = extern_kernels.convolution(buf512, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf513, (8, 208, 28, 28), (163072, 784, 28, 1))
        buf514 = reinterpret_tensor(buf501, (8, 208, 28, 28), (163072, 1, 5824, 208), 0); del buf501  # reuse
        # Source Nodes: [out_48], Original ATen: [aten.convolution]
        triton_poi_fused_cat_42.run(buf513, buf514, 1664, 784, grid=grid(1664, 784), stream=stream0)
        buf515 = buf462; del buf462  # reuse
        buf516 = buf461; del buf461  # reuse
        buf517 = buf460; del buf460  # reuse
        # Source Nodes: [out_49], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_47.run(buf514, buf515, buf516, buf517, 10192, 128, grid=grid(10192), stream=stream0)
        buf518 = buf464; del buf464  # reuse
        buf519 = empty_strided((1, 208, 1, 1), (208, 1, 208, 208), device='cuda', dtype=torch.float32)
        buf521 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_49], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_48.run(buf515, buf516, buf517, primals_612, primals_613, buf518, buf519, buf521, primals_612, primals_613, 208, 49, grid=grid(208), stream=stream0)
        del buf515
        del buf516
        del buf517
        del primals_612
        del primals_613
        buf522 = reinterpret_tensor(buf513, (8, 208, 28, 28), (163072, 1, 5824, 208), 0); del buf513  # reuse
        buf2125 = empty_strided((8, 208, 28, 28), (163072, 1, 5824, 208), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_49, out_50], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_49.run(buf514, buf518, buf519, primals_101, primals_102, buf522, buf2125, 1304576, grid=grid(1304576), stream=stream0)
        del primals_102
        # Source Nodes: [sp_79], Original ATen: [aten.convolution]
        buf523 = extern_kernels.convolution(reinterpret_tensor(buf522, (8, 52, 28, 28), (163072, 1, 5824, 208), 0), buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf523, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf524 = reinterpret_tensor(buf490, (8, 52, 28, 28), (40768, 1, 1456, 52), 0); del buf490  # reuse
        # Source Nodes: [sp_79], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf523, buf524, 416, 784, grid=grid(416, 784), stream=stream0)
        buf525 = buf494; del buf494  # reuse
        buf526 = buf493; del buf493  # reuse
        buf527 = buf492; del buf492  # reuse
        # Source Nodes: [sp_80], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf524, buf525, buf526, buf527, 2548, 128, grid=grid(2548), stream=stream0)
        buf528 = buf496; del buf496  # reuse
        buf529 = empty_strided((1, 52, 1, 1), (52, 1, 52, 52), device='cuda', dtype=torch.float32)
        buf531 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_80], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf525, buf526, buf527, primals_615, primals_616, buf528, buf529, buf531, primals_615, primals_616, 52, 49, grid=grid(52), stream=stream0)
        del primals_615
        del primals_616
        buf556 = empty((8, 208, 28, 28), device='cuda', dtype=torch.float32)
        buf532 = reinterpret_tensor(buf556, (8, 52, 28, 28), (163072, 784, 28, 1), 0)  # alias
        buf2124 = empty_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_80, sp_81], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_40.run(buf524, buf528, buf529, primals_104, primals_105, buf532, buf2124, 416, 784, grid=grid(416, 784), stream=stream0)
        del primals_105
        buf533 = reinterpret_tensor(buf523, (8, 52, 28, 28), (40768, 1, 1456, 52), 0); del buf523  # reuse
        # Source Nodes: [sp_82], Original ATen: [aten.add]
        triton_poi_fused_add_50.run(buf532, buf522, buf533, 6272, 52, grid=grid(6272, 52), stream=stream0)
        # Source Nodes: [sp_83], Original ATen: [aten.convolution]
        buf534 = extern_kernels.convolution(buf533, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf534, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf535 = empty_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_83], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf534, buf535, 416, 784, grid=grid(416, 784), stream=stream0)
        buf536 = buf527; del buf527  # reuse
        buf537 = buf526; del buf526  # reuse
        buf538 = buf525; del buf525  # reuse
        # Source Nodes: [sp_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf535, buf536, buf537, buf538, 2548, 128, grid=grid(2548), stream=stream0)
        buf539 = buf529; del buf529  # reuse
        buf540 = empty_strided((1, 52, 1, 1), (52, 1, 52, 52), device='cuda', dtype=torch.float32)
        buf542 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf536, buf537, buf538, primals_618, primals_619, buf539, buf540, buf542, primals_618, primals_619, 52, 49, grid=grid(52), stream=stream0)
        del primals_618
        del primals_619
        buf543 = reinterpret_tensor(buf556, (8, 52, 28, 28), (163072, 784, 28, 1), 40768)  # alias
        buf2123 = empty_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_84, sp_85], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_40.run(buf535, buf539, buf540, primals_107, primals_108, buf543, buf2123, 416, 784, grid=grid(416, 784), stream=stream0)
        del primals_108
        buf544 = reinterpret_tensor(buf534, (8, 52, 28, 28), (40768, 1, 1456, 52), 0); del buf534  # reuse
        # Source Nodes: [sp_86], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(buf543, buf522, buf544, 6272, 52, grid=grid(6272, 52), stream=stream0)
        # Source Nodes: [sp_87], Original ATen: [aten.convolution]
        buf545 = extern_kernels.convolution(buf544, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf545, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf546 = empty_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_87], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf545, buf546, 416, 784, grid=grid(416, 784), stream=stream0)
        buf547 = buf538; del buf538  # reuse
        buf548 = buf537; del buf537  # reuse
        buf549 = buf536; del buf536  # reuse
        # Source Nodes: [sp_88], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf546, buf547, buf548, buf549, 2548, 128, grid=grid(2548), stream=stream0)
        buf550 = buf540; del buf540  # reuse
        buf551 = empty_strided((1, 52, 1, 1), (52, 1, 52, 52), device='cuda', dtype=torch.float32)
        buf553 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_88], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf547, buf548, buf549, primals_621, primals_622, buf550, buf551, buf553, primals_621, primals_622, 52, 49, grid=grid(52), stream=stream0)
        del buf547
        del buf548
        del buf549
        del primals_621
        del primals_622
        buf554 = reinterpret_tensor(buf556, (8, 52, 28, 28), (163072, 784, 28, 1), 81536)  # alias
        buf2122 = empty_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_88, sp_89], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_40.run(buf546, buf550, buf551, primals_110, primals_111, buf554, buf2122, 416, 784, grid=grid(416, 784), stream=stream0)
        del buf551
        del primals_111
        buf555 = reinterpret_tensor(buf556, (8, 52, 28, 28), (163072, 784, 28, 1), 122304)  # alias
        # Source Nodes: [cat_59], Original ATen: [aten.cat]
        triton_poi_fused_cat_52.run(buf522, buf555, 416, 784, grid=grid(416, 784), stream=stream0)
        buf557 = empty_strided((8, 208, 28, 28), (163072, 1, 5824, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_59], Original ATen: [aten.cat]
        triton_poi_fused_cat_42.run(buf556, buf557, 1664, 784, grid=grid(1664, 784), stream=stream0)
        del buf532
        del buf543
        del buf554
        del buf555
        # Source Nodes: [out_52], Original ATen: [aten.convolution]
        buf558 = extern_kernels.convolution(buf557, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf558, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf559 = empty_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_52], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf558, buf559, 4096, 784, grid=grid(4096, 784), stream=stream0)
        buf560 = buf507; del buf507  # reuse
        buf561 = buf506; del buf506  # reuse
        buf562 = buf505; del buf505  # reuse
        # Source Nodes: [out_53], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf559, buf560, buf561, buf562, 25088, 128, grid=grid(25088), stream=stream0)
        buf563 = buf509; del buf509  # reuse
        buf564 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf566 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_53], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf560, buf561, buf562, primals_624, primals_625, buf563, buf564, buf566, primals_624, primals_625, 512, 49, grid=grid(512), stream=stream0)
        del buf560
        del buf561
        del buf562
        del primals_624
        del primals_625
        buf567 = reinterpret_tensor(buf558, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf558  # reuse
        # Source Nodes: [out_53, out_54, shortcut_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_53.run(buf559, buf563, buf564, primals_113, primals_114, buf512, buf567, 3211264, grid=grid(3211264), stream=stream0)
        del buf564
        del primals_114
        # Source Nodes: [out_56], Original ATen: [aten.convolution]
        buf568 = extern_kernels.convolution(buf567, primals_115, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf568, (8, 416, 28, 28), (326144, 784, 28, 1))
        buf569 = reinterpret_tensor(buf322, (8, 416, 28, 28), (326144, 1, 11648, 416), 0); del buf322  # reuse
        # Source Nodes: [out_56], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf568, buf569, 3328, 784, grid=grid(3328, 784), stream=stream0)
        buf570 = reinterpret_tensor(buf271, (1, 416, 1, 1, 49), (20384, 1, 20384, 20384, 416), 0); del buf271  # reuse
        buf571 = reinterpret_tensor(buf270, (1, 416, 1, 1, 49), (20384, 1, 20384, 20384, 416), 0); del buf270  # reuse
        buf572 = reinterpret_tensor(buf269, (1, 416, 1, 1, 49), (20384, 1, 20384, 20384, 416), 0); del buf269  # reuse
        # Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf569, buf570, buf571, buf572, 20384, 128, grid=grid(20384), stream=stream0)
        buf573 = reinterpret_tensor(buf344, (1, 416, 1, 1), (416, 1, 416, 416), 0); del buf344  # reuse
        buf574 = reinterpret_tensor(buf343, (1, 416, 1, 1), (416, 1, 416, 416), 0); del buf343  # reuse
        buf576 = reinterpret_tensor(buf342, (416, ), (1, ), 0); del buf342  # reuse
        # Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_56.run(buf570, buf571, buf572, primals_627, primals_628, buf573, buf574, buf576, primals_627, primals_628, 416, 49, grid=grid(416), stream=stream0)
        del buf570
        del buf571
        del buf572
        del primals_627
        del primals_628
        buf577 = reinterpret_tensor(buf568, (8, 416, 28, 28), (326144, 1, 11648, 416), 0); del buf568  # reuse
        buf2121 = empty_strided((8, 416, 28, 28), (326144, 1, 11648, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_57, out_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_57.run(buf569, buf573, buf574, primals_116, primals_117, buf577, buf2121, 2609152, grid=grid(2609152), stream=stream0)
        del primals_117
        # Source Nodes: [sp_92], Original ATen: [aten.convolution]
        buf578 = extern_kernels.convolution(reinterpret_tensor(buf577, (8, 104, 28, 28), (326144, 1, 11648, 416), 0), buf22, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf578, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf579 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_92], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf578, buf579, 832, 196, grid=grid(832, 196), stream=stream0)
        buf580 = empty_strided((1, 104, 1, 1, 13), (1352, 1, 1352, 1352, 104), device='cuda', dtype=torch.float32)
        buf581 = empty_strided((1, 104, 1, 1, 13), (1352, 1, 1352, 1352, 104), device='cuda', dtype=torch.float32)
        buf582 = empty_strided((1, 104, 1, 1, 13), (1352, 1, 1352, 1352, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_93], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf579, buf580, buf581, buf582, 1352, 121, grid=grid(1352), stream=stream0)
        buf583 = buf276; del buf276  # reuse
        buf584 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf586 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_93], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf580, buf581, buf582, primals_630, primals_631, buf583, buf584, buf586, primals_630, primals_631, 104, 13, grid=grid(104), stream=stream0)
        del primals_630
        del primals_631
        buf609 = reinterpret_tensor(buf308, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf308  # reuse
        buf587 = reinterpret_tensor(buf609, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2120 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_93, sp_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf579, buf583, buf584, primals_119, primals_120, buf587, buf2120, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_120
        # Source Nodes: [sp_96], Original ATen: [aten.convolution]
        buf588 = extern_kernels.convolution(reinterpret_tensor(buf577, (8, 104, 28, 28), (326144, 1, 11648, 416), 104), buf23, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf588, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf589 = reinterpret_tensor(buf578, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf578  # reuse
        # Source Nodes: [sp_96], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf588, buf589, 832, 196, grid=grid(832, 196), stream=stream0)
        buf590 = buf582; del buf582  # reuse
        buf591 = buf581; del buf581  # reuse
        buf592 = buf580; del buf580  # reuse
        # Source Nodes: [sp_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf589, buf590, buf591, buf592, 1352, 121, grid=grid(1352), stream=stream0)
        buf593 = buf584; del buf584  # reuse
        buf594 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf596 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf590, buf591, buf592, primals_633, primals_634, buf593, buf594, buf596, primals_633, primals_634, 104, 13, grid=grid(104), stream=stream0)
        del primals_633
        del primals_634
        buf597 = reinterpret_tensor(buf609, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2119 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_97, sp_98], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf589, buf593, buf594, primals_122, primals_123, buf597, buf2119, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_123
        # Source Nodes: [sp_100], Original ATen: [aten.convolution]
        buf598 = extern_kernels.convolution(reinterpret_tensor(buf577, (8, 104, 28, 28), (326144, 1, 11648, 416), 208), buf24, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf598, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf599 = reinterpret_tensor(buf588, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf588  # reuse
        # Source Nodes: [sp_100], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf598, buf599, 832, 196, grid=grid(832, 196), stream=stream0)
        buf600 = buf592; del buf592  # reuse
        buf601 = buf591; del buf591  # reuse
        buf602 = buf590; del buf590  # reuse
        # Source Nodes: [sp_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf599, buf600, buf601, buf602, 1352, 121, grid=grid(1352), stream=stream0)
        buf603 = buf594; del buf594  # reuse
        buf604 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf606 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf600, buf601, buf602, primals_636, primals_637, buf603, buf604, buf606, primals_636, primals_637, 104, 13, grid=grid(104), stream=stream0)
        del primals_636
        del primals_637
        buf607 = reinterpret_tensor(buf609, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2118 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_101, sp_102], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf599, buf603, buf604, primals_125, primals_126, buf607, buf2118, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_126
        buf608 = reinterpret_tensor(buf609, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [getattr_l__mod___layer3___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_62.run(buf577, buf608, 832, 196, grid=grid(832, 196), stream=stream0)
        buf610 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_58], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf609, buf610, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf587
        del buf597
        del buf607
        del buf608
        # Source Nodes: [out_60], Original ATen: [aten.convolution]
        buf611 = extern_kernels.convolution(buf610, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf611, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf612 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_60], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf611, buf612, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf613 = empty_strided((1, 1024, 1, 1, 13), (13312, 1, 13312, 13312, 1024), device='cuda', dtype=torch.float32)
        buf614 = empty_strided((1, 1024, 1, 1, 13), (13312, 1, 13312, 13312, 1024), device='cuda', dtype=torch.float32)
        buf615 = empty_strided((1, 1024, 1, 1, 13), (13312, 1, 13312, 13312, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf612, buf613, buf614, buf615, 13312, 121, grid=grid(13312), stream=stream0)
        buf616 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf617 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf619 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf613, buf614, buf615, primals_639, primals_640, buf616, buf617, buf619, primals_639, primals_640, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_639
        del primals_640
        # Source Nodes: [getattr_l__mod___layer3___0___downsample_0], Original ATen: [aten.convolution]
        buf620 = extern_kernels.convolution(buf567, primals_130, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf620, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf621 = reinterpret_tensor(buf611, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf611  # reuse
        # Source Nodes: [getattr_l__mod___layer3___0___downsample_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf620, buf621, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf622 = buf615; del buf615  # reuse
        buf623 = buf614; del buf614  # reuse
        buf624 = buf613; del buf613  # reuse
        # Source Nodes: [shortcut_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf621, buf622, buf623, buf624, 13312, 121, grid=grid(13312), stream=stream0)
        buf625 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf626 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf628 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf622, buf623, buf624, primals_642, primals_643, buf625, buf626, buf628, primals_642, primals_643, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_642
        del primals_643
        buf629 = reinterpret_tensor(buf620, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf620  # reuse
        buf630 = buf629; del buf629  # reuse
        # Source Nodes: [out_61, out_62, shortcut_10, shortcut_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_67.run(buf630, buf612, buf616, buf617, primals_128, primals_129, buf621, buf625, buf626, primals_131, primals_132, 1605632, grid=grid(1605632), stream=stream0)
        del primals_129
        del primals_132
        # Source Nodes: [out_64], Original ATen: [aten.convolution]
        buf631 = extern_kernels.convolution(buf630, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf631, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf632 = reinterpret_tensor(buf609, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf609  # reuse
        # Source Nodes: [out_64], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf631, buf632, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf633 = empty_strided((1, 416, 1, 1, 13), (5408, 1, 5408, 5408, 416), device='cuda', dtype=torch.float32)
        buf634 = empty_strided((1, 416, 1, 1, 13), (5408, 1, 5408, 5408, 416), device='cuda', dtype=torch.float32)
        buf635 = empty_strided((1, 416, 1, 1, 13), (5408, 1, 5408, 5408, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_65], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf632, buf633, buf634, buf635, 5408, 121, grid=grid(5408), stream=stream0)
        buf636 = buf574; del buf574  # reuse
        buf637 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf639 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_65], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf633, buf634, buf635, primals_645, primals_646, buf636, buf637, buf639, primals_645, primals_646, 416, 13, grid=grid(416), stream=stream0)
        del primals_645
        del primals_646
        buf640 = reinterpret_tensor(buf631, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf631  # reuse
        buf2117 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_65, out_66], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf632, buf636, buf637, primals_134, primals_135, buf640, buf2117, 652288, grid=grid(652288), stream=stream0)
        del primals_135
        # Source Nodes: [sp_105], Original ATen: [aten.convolution]
        buf641 = extern_kernels.convolution(reinterpret_tensor(buf640, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf641, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf642 = reinterpret_tensor(buf598, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf598  # reuse
        # Source Nodes: [sp_105], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf641, buf642, 832, 196, grid=grid(832, 196), stream=stream0)
        buf643 = buf602; del buf602  # reuse
        buf644 = buf601; del buf601  # reuse
        buf645 = buf600; del buf600  # reuse
        # Source Nodes: [sp_106], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf642, buf643, buf644, buf645, 1352, 121, grid=grid(1352), stream=stream0)
        buf646 = buf604; del buf604  # reuse
        buf647 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf649 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_106], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf643, buf644, buf645, primals_648, primals_649, buf646, buf647, buf649, primals_648, primals_649, 104, 13, grid=grid(104), stream=stream0)
        del primals_648
        del primals_649
        buf674 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf650 = reinterpret_tensor(buf674, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2116 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_106, sp_107], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf642, buf646, buf647, primals_137, primals_138, buf650, buf2116, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_138
        buf651 = reinterpret_tensor(buf641, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf641  # reuse
        # Source Nodes: [sp_108], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf650, buf640, buf651, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_109], Original ATen: [aten.convolution]
        buf652 = extern_kernels.convolution(buf651, buf26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf652, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf653 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_109], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf652, buf653, 832, 196, grid=grid(832, 196), stream=stream0)
        buf654 = buf645; del buf645  # reuse
        buf655 = buf644; del buf644  # reuse
        buf656 = buf643; del buf643  # reuse
        # Source Nodes: [sp_110], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf653, buf654, buf655, buf656, 1352, 121, grid=grid(1352), stream=stream0)
        buf657 = buf647; del buf647  # reuse
        buf658 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf660 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_110], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf654, buf655, buf656, primals_651, primals_652, buf657, buf658, buf660, primals_651, primals_652, 104, 13, grid=grid(104), stream=stream0)
        del primals_651
        del primals_652
        buf661 = reinterpret_tensor(buf674, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2115 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_110, sp_111], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf653, buf657, buf658, primals_140, primals_141, buf661, buf2115, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_141
        buf662 = reinterpret_tensor(buf652, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf652  # reuse
        # Source Nodes: [sp_112], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf661, buf640, buf662, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_113], Original ATen: [aten.convolution]
        buf663 = extern_kernels.convolution(buf662, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf663, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf664 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_113], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf663, buf664, 832, 196, grid=grid(832, 196), stream=stream0)
        buf665 = buf656; del buf656  # reuse
        buf666 = buf655; del buf655  # reuse
        buf667 = buf654; del buf654  # reuse
        # Source Nodes: [sp_114], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf664, buf665, buf666, buf667, 1352, 121, grid=grid(1352), stream=stream0)
        buf668 = buf658; del buf658  # reuse
        buf669 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf671 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_114], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf665, buf666, buf667, primals_654, primals_655, buf668, buf669, buf671, primals_654, primals_655, 104, 13, grid=grid(104), stream=stream0)
        del primals_654
        del primals_655
        buf672 = reinterpret_tensor(buf674, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2114 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_114, sp_115], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf664, buf668, buf669, primals_143, primals_144, buf672, buf2114, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_144
        buf673 = reinterpret_tensor(buf674, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_57], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf640, buf673, 832, 196, grid=grid(832, 196), stream=stream0)
        buf675 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_57], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf674, buf675, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf650
        del buf661
        del buf672
        del buf673
        # Source Nodes: [out_68], Original ATen: [aten.convolution]
        buf676 = extern_kernels.convolution(buf675, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf676, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf677 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_68], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf676, buf677, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf678 = buf624; del buf624  # reuse
        buf679 = buf623; del buf623  # reuse
        buf680 = buf622; del buf622  # reuse
        # Source Nodes: [out_69], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf677, buf678, buf679, buf680, 13312, 121, grid=grid(13312), stream=stream0)
        buf681 = buf626; del buf626  # reuse
        buf682 = buf617; del buf617  # reuse
        buf684 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_69], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf678, buf679, buf680, primals_657, primals_658, buf681, buf682, buf684, primals_657, primals_658, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_657
        del primals_658
        buf685 = reinterpret_tensor(buf676, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf676  # reuse
        # Source Nodes: [out_69, out_70, shortcut_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf677, buf681, buf682, primals_146, primals_147, buf630, buf685, 1605632, grid=grid(1605632), stream=stream0)
        del primals_147
        # Source Nodes: [out_72], Original ATen: [aten.convolution]
        buf686 = extern_kernels.convolution(buf685, primals_148, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf686, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf687 = reinterpret_tensor(buf674, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf674  # reuse
        # Source Nodes: [out_72], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf686, buf687, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf688 = buf635; del buf635  # reuse
        buf689 = buf634; del buf634  # reuse
        buf690 = buf633; del buf633  # reuse
        # Source Nodes: [out_73], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf687, buf688, buf689, buf690, 5408, 121, grid=grid(5408), stream=stream0)
        buf691 = buf637; del buf637  # reuse
        buf692 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf694 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_73], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf688, buf689, buf690, primals_660, primals_661, buf691, buf692, buf694, primals_660, primals_661, 416, 13, grid=grid(416), stream=stream0)
        del primals_660
        del primals_661
        buf695 = reinterpret_tensor(buf686, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf686  # reuse
        buf2113 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_73, out_74], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf687, buf691, buf692, primals_149, primals_150, buf695, buf2113, 652288, grid=grid(652288), stream=stream0)
        del primals_150
        # Source Nodes: [sp_118], Original ATen: [aten.convolution]
        buf696 = extern_kernels.convolution(reinterpret_tensor(buf695, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf696, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf697 = reinterpret_tensor(buf663, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf663  # reuse
        # Source Nodes: [sp_118], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf696, buf697, 832, 196, grid=grid(832, 196), stream=stream0)
        buf698 = buf667; del buf667  # reuse
        buf699 = buf666; del buf666  # reuse
        buf700 = buf665; del buf665  # reuse
        # Source Nodes: [sp_119], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf697, buf698, buf699, buf700, 1352, 121, grid=grid(1352), stream=stream0)
        buf701 = buf669; del buf669  # reuse
        buf702 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf704 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_119], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf698, buf699, buf700, primals_663, primals_664, buf701, buf702, buf704, primals_663, primals_664, 104, 13, grid=grid(104), stream=stream0)
        del primals_663
        del primals_664
        buf729 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf705 = reinterpret_tensor(buf729, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2112 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_119, sp_120], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf697, buf701, buf702, primals_152, primals_153, buf705, buf2112, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_153
        buf706 = reinterpret_tensor(buf696, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf696  # reuse
        # Source Nodes: [sp_121], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf705, buf695, buf706, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_122], Original ATen: [aten.convolution]
        buf707 = extern_kernels.convolution(buf706, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf707, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf708 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_122], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf707, buf708, 832, 196, grid=grid(832, 196), stream=stream0)
        buf709 = buf700; del buf700  # reuse
        buf710 = buf699; del buf699  # reuse
        buf711 = buf698; del buf698  # reuse
        # Source Nodes: [sp_123], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf708, buf709, buf710, buf711, 1352, 121, grid=grid(1352), stream=stream0)
        buf712 = buf702; del buf702  # reuse
        buf713 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf715 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_123], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf709, buf710, buf711, primals_666, primals_667, buf712, buf713, buf715, primals_666, primals_667, 104, 13, grid=grid(104), stream=stream0)
        del primals_666
        del primals_667
        buf716 = reinterpret_tensor(buf729, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2111 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_123, sp_124], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf708, buf712, buf713, primals_155, primals_156, buf716, buf2111, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_156
        buf717 = reinterpret_tensor(buf707, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf707  # reuse
        # Source Nodes: [sp_125], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf716, buf695, buf717, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_126], Original ATen: [aten.convolution]
        buf718 = extern_kernels.convolution(buf717, buf30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf718, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf719 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_126], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf718, buf719, 832, 196, grid=grid(832, 196), stream=stream0)
        buf720 = buf711; del buf711  # reuse
        buf721 = buf710; del buf710  # reuse
        buf722 = buf709; del buf709  # reuse
        # Source Nodes: [sp_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf719, buf720, buf721, buf722, 1352, 121, grid=grid(1352), stream=stream0)
        buf723 = buf713; del buf713  # reuse
        buf724 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf726 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf720, buf721, buf722, primals_669, primals_670, buf723, buf724, buf726, primals_669, primals_670, 104, 13, grid=grid(104), stream=stream0)
        del primals_669
        del primals_670
        buf727 = reinterpret_tensor(buf729, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2110 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_127, sp_128], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf719, buf723, buf724, primals_158, primals_159, buf727, buf2110, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_159
        buf728 = reinterpret_tensor(buf729, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_56], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf695, buf728, 832, 196, grid=grid(832, 196), stream=stream0)
        buf730 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_56], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf729, buf730, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf705
        del buf716
        del buf727
        del buf728
        # Source Nodes: [out_76], Original ATen: [aten.convolution]
        buf731 = extern_kernels.convolution(buf730, primals_160, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf731, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf732 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_76], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf731, buf732, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf733 = buf680; del buf680  # reuse
        buf734 = buf679; del buf679  # reuse
        buf735 = buf678; del buf678  # reuse
        # Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf732, buf733, buf734, buf735, 13312, 121, grid=grid(13312), stream=stream0)
        buf736 = buf682; del buf682  # reuse
        buf737 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf739 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf733, buf734, buf735, primals_672, primals_673, buf736, buf737, buf739, primals_672, primals_673, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_672
        del primals_673
        buf740 = reinterpret_tensor(buf731, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf731  # reuse
        # Source Nodes: [out_77, out_78, shortcut_13], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf732, buf736, buf737, primals_161, primals_162, buf685, buf740, 1605632, grid=grid(1605632), stream=stream0)
        del primals_162
        # Source Nodes: [out_80], Original ATen: [aten.convolution]
        buf741 = extern_kernels.convolution(buf740, primals_163, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf741, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf742 = reinterpret_tensor(buf729, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf729  # reuse
        # Source Nodes: [out_80], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf741, buf742, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf743 = buf690; del buf690  # reuse
        buf744 = buf689; del buf689  # reuse
        buf745 = buf688; del buf688  # reuse
        # Source Nodes: [out_81], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf742, buf743, buf744, buf745, 5408, 121, grid=grid(5408), stream=stream0)
        buf746 = buf692; del buf692  # reuse
        buf747 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf749 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_81], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf743, buf744, buf745, primals_675, primals_676, buf746, buf747, buf749, primals_675, primals_676, 416, 13, grid=grid(416), stream=stream0)
        del primals_675
        del primals_676
        buf750 = reinterpret_tensor(buf741, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf741  # reuse
        buf2109 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_81, out_82], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf742, buf746, buf747, primals_164, primals_165, buf750, buf2109, 652288, grid=grid(652288), stream=stream0)
        del primals_165
        # Source Nodes: [sp_131], Original ATen: [aten.convolution]
        buf751 = extern_kernels.convolution(reinterpret_tensor(buf750, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf751, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf752 = reinterpret_tensor(buf718, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf718  # reuse
        # Source Nodes: [sp_131], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf751, buf752, 832, 196, grid=grid(832, 196), stream=stream0)
        buf753 = buf722; del buf722  # reuse
        buf754 = buf721; del buf721  # reuse
        buf755 = buf720; del buf720  # reuse
        # Source Nodes: [sp_132], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf752, buf753, buf754, buf755, 1352, 121, grid=grid(1352), stream=stream0)
        buf756 = buf724; del buf724  # reuse
        buf757 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf759 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_132], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf753, buf754, buf755, primals_678, primals_679, buf756, buf757, buf759, primals_678, primals_679, 104, 13, grid=grid(104), stream=stream0)
        del primals_678
        del primals_679
        buf784 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf760 = reinterpret_tensor(buf784, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2108 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_132, sp_133], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf752, buf756, buf757, primals_167, primals_168, buf760, buf2108, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_168
        buf761 = reinterpret_tensor(buf751, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf751  # reuse
        # Source Nodes: [sp_134], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf760, buf750, buf761, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_135], Original ATen: [aten.convolution]
        buf762 = extern_kernels.convolution(buf761, buf32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf762, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf763 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_135], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf762, buf763, 832, 196, grid=grid(832, 196), stream=stream0)
        buf764 = buf755; del buf755  # reuse
        buf765 = buf754; del buf754  # reuse
        buf766 = buf753; del buf753  # reuse
        # Source Nodes: [sp_136], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf763, buf764, buf765, buf766, 1352, 121, grid=grid(1352), stream=stream0)
        buf767 = buf757; del buf757  # reuse
        buf768 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf770 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_136], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf764, buf765, buf766, primals_681, primals_682, buf767, buf768, buf770, primals_681, primals_682, 104, 13, grid=grid(104), stream=stream0)
        del primals_681
        del primals_682
        buf771 = reinterpret_tensor(buf784, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2107 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_136, sp_137], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf763, buf767, buf768, primals_170, primals_171, buf771, buf2107, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_171
        buf772 = reinterpret_tensor(buf762, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf762  # reuse
        # Source Nodes: [sp_138], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf771, buf750, buf772, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_139], Original ATen: [aten.convolution]
        buf773 = extern_kernels.convolution(buf772, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf773, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf774 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_139], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf773, buf774, 832, 196, grid=grid(832, 196), stream=stream0)
        buf775 = buf766; del buf766  # reuse
        buf776 = buf765; del buf765  # reuse
        buf777 = buf764; del buf764  # reuse
        # Source Nodes: [sp_140], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf774, buf775, buf776, buf777, 1352, 121, grid=grid(1352), stream=stream0)
        buf778 = buf768; del buf768  # reuse
        buf779 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf781 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_140], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf775, buf776, buf777, primals_684, primals_685, buf778, buf779, buf781, primals_684, primals_685, 104, 13, grid=grid(104), stream=stream0)
        del primals_684
        del primals_685
        buf782 = reinterpret_tensor(buf784, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2106 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_140, sp_141], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf774, buf778, buf779, primals_173, primals_174, buf782, buf2106, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_174
        buf783 = reinterpret_tensor(buf784, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_55], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf750, buf783, 832, 196, grid=grid(832, 196), stream=stream0)
        buf785 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_55], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf784, buf785, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf760
        del buf771
        del buf782
        del buf783
        # Source Nodes: [out_84], Original ATen: [aten.convolution]
        buf786 = extern_kernels.convolution(buf785, primals_175, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf786, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf787 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_84], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf786, buf787, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf788 = buf735; del buf735  # reuse
        buf789 = buf734; del buf734  # reuse
        buf790 = buf733; del buf733  # reuse
        # Source Nodes: [out_85], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf787, buf788, buf789, buf790, 13312, 121, grid=grid(13312), stream=stream0)
        buf791 = buf737; del buf737  # reuse
        buf792 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf794 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_85], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf788, buf789, buf790, primals_687, primals_688, buf791, buf792, buf794, primals_687, primals_688, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_687
        del primals_688
        buf795 = reinterpret_tensor(buf786, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf786  # reuse
        # Source Nodes: [out_85, out_86, shortcut_14], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf787, buf791, buf792, primals_176, primals_177, buf740, buf795, 1605632, grid=grid(1605632), stream=stream0)
        del primals_177
        # Source Nodes: [out_88], Original ATen: [aten.convolution]
        buf796 = extern_kernels.convolution(buf795, primals_178, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf796, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf797 = reinterpret_tensor(buf784, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf784  # reuse
        # Source Nodes: [out_88], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf796, buf797, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf798 = buf745; del buf745  # reuse
        buf799 = buf744; del buf744  # reuse
        buf800 = buf743; del buf743  # reuse
        # Source Nodes: [out_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf797, buf798, buf799, buf800, 5408, 121, grid=grid(5408), stream=stream0)
        buf801 = buf747; del buf747  # reuse
        buf802 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf804 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf798, buf799, buf800, primals_690, primals_691, buf801, buf802, buf804, primals_690, primals_691, 416, 13, grid=grid(416), stream=stream0)
        del primals_690
        del primals_691
        buf805 = reinterpret_tensor(buf796, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf796  # reuse
        buf2105 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_89, out_90], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf797, buf801, buf802, primals_179, primals_180, buf805, buf2105, 652288, grid=grid(652288), stream=stream0)
        del primals_180
        # Source Nodes: [sp_144], Original ATen: [aten.convolution]
        buf806 = extern_kernels.convolution(reinterpret_tensor(buf805, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf806, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf807 = reinterpret_tensor(buf773, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf773  # reuse
        # Source Nodes: [sp_144], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf806, buf807, 832, 196, grid=grid(832, 196), stream=stream0)
        buf808 = buf777; del buf777  # reuse
        buf809 = buf776; del buf776  # reuse
        buf810 = buf775; del buf775  # reuse
        # Source Nodes: [sp_145], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf807, buf808, buf809, buf810, 1352, 121, grid=grid(1352), stream=stream0)
        buf811 = buf779; del buf779  # reuse
        buf812 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf814 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_145], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf808, buf809, buf810, primals_693, primals_694, buf811, buf812, buf814, primals_693, primals_694, 104, 13, grid=grid(104), stream=stream0)
        del primals_693
        del primals_694
        buf839 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf815 = reinterpret_tensor(buf839, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2104 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_145, sp_146], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf807, buf811, buf812, primals_182, primals_183, buf815, buf2104, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_183
        buf816 = reinterpret_tensor(buf806, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf806  # reuse
        # Source Nodes: [sp_147], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf815, buf805, buf816, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_148], Original ATen: [aten.convolution]
        buf817 = extern_kernels.convolution(buf816, buf35, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf817, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf818 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_148], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf817, buf818, 832, 196, grid=grid(832, 196), stream=stream0)
        buf819 = buf810; del buf810  # reuse
        buf820 = buf809; del buf809  # reuse
        buf821 = buf808; del buf808  # reuse
        # Source Nodes: [sp_149], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf818, buf819, buf820, buf821, 1352, 121, grid=grid(1352), stream=stream0)
        buf822 = buf812; del buf812  # reuse
        buf823 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf825 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_149], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf819, buf820, buf821, primals_696, primals_697, buf822, buf823, buf825, primals_696, primals_697, 104, 13, grid=grid(104), stream=stream0)
        del primals_696
        del primals_697
        buf826 = reinterpret_tensor(buf839, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2103 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_149, sp_150], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf818, buf822, buf823, primals_185, primals_186, buf826, buf2103, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_186
        buf827 = reinterpret_tensor(buf817, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf817  # reuse
        # Source Nodes: [sp_151], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf826, buf805, buf827, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_152], Original ATen: [aten.convolution]
        buf828 = extern_kernels.convolution(buf827, buf36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf828, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf829 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_152], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf828, buf829, 832, 196, grid=grid(832, 196), stream=stream0)
        buf830 = buf821; del buf821  # reuse
        buf831 = buf820; del buf820  # reuse
        buf832 = buf819; del buf819  # reuse
        # Source Nodes: [sp_153], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf829, buf830, buf831, buf832, 1352, 121, grid=grid(1352), stream=stream0)
        buf833 = buf823; del buf823  # reuse
        buf834 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf836 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_153], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf830, buf831, buf832, primals_699, primals_700, buf833, buf834, buf836, primals_699, primals_700, 104, 13, grid=grid(104), stream=stream0)
        del primals_699
        del primals_700
        buf837 = reinterpret_tensor(buf839, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2102 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_153, sp_154], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf829, buf833, buf834, primals_188, primals_189, buf837, buf2102, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_189
        buf838 = reinterpret_tensor(buf839, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_54], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf805, buf838, 832, 196, grid=grid(832, 196), stream=stream0)
        buf840 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_54], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf839, buf840, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf815
        del buf826
        del buf837
        del buf838
        # Source Nodes: [out_92], Original ATen: [aten.convolution]
        buf841 = extern_kernels.convolution(buf840, primals_190, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf841, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf842 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_92], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf841, buf842, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf843 = buf790; del buf790  # reuse
        buf844 = buf789; del buf789  # reuse
        buf845 = buf788; del buf788  # reuse
        # Source Nodes: [out_93], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf842, buf843, buf844, buf845, 13312, 121, grid=grid(13312), stream=stream0)
        buf846 = buf792; del buf792  # reuse
        buf847 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf849 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_93], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf843, buf844, buf845, primals_702, primals_703, buf846, buf847, buf849, primals_702, primals_703, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_702
        del primals_703
        buf850 = reinterpret_tensor(buf841, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf841  # reuse
        # Source Nodes: [out_93, out_94, shortcut_15], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf842, buf846, buf847, primals_191, primals_192, buf795, buf850, 1605632, grid=grid(1605632), stream=stream0)
        del primals_192
        # Source Nodes: [out_96], Original ATen: [aten.convolution]
        buf851 = extern_kernels.convolution(buf850, primals_193, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf851, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf852 = reinterpret_tensor(buf839, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf839  # reuse
        # Source Nodes: [out_96], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf851, buf852, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf853 = buf800; del buf800  # reuse
        buf854 = buf799; del buf799  # reuse
        buf855 = buf798; del buf798  # reuse
        # Source Nodes: [out_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf852, buf853, buf854, buf855, 5408, 121, grid=grid(5408), stream=stream0)
        buf856 = buf802; del buf802  # reuse
        buf857 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf859 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf853, buf854, buf855, primals_705, primals_706, buf856, buf857, buf859, primals_705, primals_706, 416, 13, grid=grid(416), stream=stream0)
        del primals_705
        del primals_706
        buf860 = reinterpret_tensor(buf851, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf851  # reuse
        buf2101 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_97, out_98], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf852, buf856, buf857, primals_194, primals_195, buf860, buf2101, 652288, grid=grid(652288), stream=stream0)
        del primals_195
        # Source Nodes: [sp_157], Original ATen: [aten.convolution]
        buf861 = extern_kernels.convolution(reinterpret_tensor(buf860, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf861, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf862 = reinterpret_tensor(buf828, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf828  # reuse
        # Source Nodes: [sp_157], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf861, buf862, 832, 196, grid=grid(832, 196), stream=stream0)
        buf863 = buf832; del buf832  # reuse
        buf864 = buf831; del buf831  # reuse
        buf865 = buf830; del buf830  # reuse
        # Source Nodes: [sp_158], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf862, buf863, buf864, buf865, 1352, 121, grid=grid(1352), stream=stream0)
        buf866 = buf834; del buf834  # reuse
        buf867 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf869 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_158], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf863, buf864, buf865, primals_708, primals_709, buf866, buf867, buf869, primals_708, primals_709, 104, 13, grid=grid(104), stream=stream0)
        del primals_708
        del primals_709
        buf894 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf870 = reinterpret_tensor(buf894, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2100 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_158, sp_159], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf862, buf866, buf867, primals_197, primals_198, buf870, buf2100, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_198
        buf871 = reinterpret_tensor(buf861, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf861  # reuse
        # Source Nodes: [sp_160], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf870, buf860, buf871, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_161], Original ATen: [aten.convolution]
        buf872 = extern_kernels.convolution(buf871, buf38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf872, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf873 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_161], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf872, buf873, 832, 196, grid=grid(832, 196), stream=stream0)
        buf874 = buf865; del buf865  # reuse
        buf875 = buf864; del buf864  # reuse
        buf876 = buf863; del buf863  # reuse
        # Source Nodes: [sp_162], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf873, buf874, buf875, buf876, 1352, 121, grid=grid(1352), stream=stream0)
        buf877 = buf867; del buf867  # reuse
        buf878 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf880 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_162], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf874, buf875, buf876, primals_711, primals_712, buf877, buf878, buf880, primals_711, primals_712, 104, 13, grid=grid(104), stream=stream0)
        del primals_711
        del primals_712
        buf881 = reinterpret_tensor(buf894, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2099 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_162, sp_163], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf873, buf877, buf878, primals_200, primals_201, buf881, buf2099, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_201
        buf882 = reinterpret_tensor(buf872, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf872  # reuse
        # Source Nodes: [sp_164], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf881, buf860, buf882, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_165], Original ATen: [aten.convolution]
        buf883 = extern_kernels.convolution(buf882, buf39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf883, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf884 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_165], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf883, buf884, 832, 196, grid=grid(832, 196), stream=stream0)
        buf885 = buf876; del buf876  # reuse
        buf886 = buf875; del buf875  # reuse
        buf887 = buf874; del buf874  # reuse
        # Source Nodes: [sp_166], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf884, buf885, buf886, buf887, 1352, 121, grid=grid(1352), stream=stream0)
        buf888 = buf878; del buf878  # reuse
        buf889 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf891 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_166], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf885, buf886, buf887, primals_714, primals_715, buf888, buf889, buf891, primals_714, primals_715, 104, 13, grid=grid(104), stream=stream0)
        del primals_714
        del primals_715
        buf892 = reinterpret_tensor(buf894, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2098 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_166, sp_167], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf884, buf888, buf889, primals_203, primals_204, buf892, buf2098, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_204
        buf893 = reinterpret_tensor(buf894, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_53], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf860, buf893, 832, 196, grid=grid(832, 196), stream=stream0)
        buf895 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_53], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf894, buf895, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf870
        del buf881
        del buf892
        del buf893
        # Source Nodes: [out_100], Original ATen: [aten.convolution]
        buf896 = extern_kernels.convolution(buf895, primals_205, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf896, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf897 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_100], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf896, buf897, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf898 = buf845; del buf845  # reuse
        buf899 = buf844; del buf844  # reuse
        buf900 = buf843; del buf843  # reuse
        # Source Nodes: [out_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf897, buf898, buf899, buf900, 13312, 121, grid=grid(13312), stream=stream0)
        buf901 = buf847; del buf847  # reuse
        buf902 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf904 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf898, buf899, buf900, primals_717, primals_718, buf901, buf902, buf904, primals_717, primals_718, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_717
        del primals_718
        buf905 = reinterpret_tensor(buf896, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf896  # reuse
        # Source Nodes: [out_101, out_102, shortcut_16], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf897, buf901, buf902, primals_206, primals_207, buf850, buf905, 1605632, grid=grid(1605632), stream=stream0)
        del primals_207
        # Source Nodes: [out_104], Original ATen: [aten.convolution]
        buf906 = extern_kernels.convolution(buf905, primals_208, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf906, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf907 = reinterpret_tensor(buf894, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf894  # reuse
        # Source Nodes: [out_104], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf906, buf907, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf908 = buf855; del buf855  # reuse
        buf909 = buf854; del buf854  # reuse
        buf910 = buf853; del buf853  # reuse
        # Source Nodes: [out_105], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf907, buf908, buf909, buf910, 5408, 121, grid=grid(5408), stream=stream0)
        buf911 = buf857; del buf857  # reuse
        buf912 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf914 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_105], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf908, buf909, buf910, primals_720, primals_721, buf911, buf912, buf914, primals_720, primals_721, 416, 13, grid=grid(416), stream=stream0)
        del primals_720
        del primals_721
        buf915 = reinterpret_tensor(buf906, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf906  # reuse
        buf2097 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_105, out_106], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf907, buf911, buf912, primals_209, primals_210, buf915, buf2097, 652288, grid=grid(652288), stream=stream0)
        del primals_210
        # Source Nodes: [sp_170], Original ATen: [aten.convolution]
        buf916 = extern_kernels.convolution(reinterpret_tensor(buf915, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf916, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf917 = reinterpret_tensor(buf883, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf883  # reuse
        # Source Nodes: [sp_170], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf916, buf917, 832, 196, grid=grid(832, 196), stream=stream0)
        buf918 = buf887; del buf887  # reuse
        buf919 = buf886; del buf886  # reuse
        buf920 = buf885; del buf885  # reuse
        # Source Nodes: [sp_171], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf917, buf918, buf919, buf920, 1352, 121, grid=grid(1352), stream=stream0)
        buf921 = buf889; del buf889  # reuse
        buf922 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf924 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_171], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf918, buf919, buf920, primals_723, primals_724, buf921, buf922, buf924, primals_723, primals_724, 104, 13, grid=grid(104), stream=stream0)
        del primals_723
        del primals_724
        buf949 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf925 = reinterpret_tensor(buf949, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2096 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_171, sp_172], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf917, buf921, buf922, primals_212, primals_213, buf925, buf2096, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_213
        buf926 = reinterpret_tensor(buf916, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf916  # reuse
        # Source Nodes: [sp_173], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf925, buf915, buf926, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_174], Original ATen: [aten.convolution]
        buf927 = extern_kernels.convolution(buf926, buf41, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf927, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf928 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_174], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf927, buf928, 832, 196, grid=grid(832, 196), stream=stream0)
        buf929 = buf920; del buf920  # reuse
        buf930 = buf919; del buf919  # reuse
        buf931 = buf918; del buf918  # reuse
        # Source Nodes: [sp_175], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf928, buf929, buf930, buf931, 1352, 121, grid=grid(1352), stream=stream0)
        buf932 = buf922; del buf922  # reuse
        buf933 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf935 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_175], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf929, buf930, buf931, primals_726, primals_727, buf932, buf933, buf935, primals_726, primals_727, 104, 13, grid=grid(104), stream=stream0)
        del primals_726
        del primals_727
        buf936 = reinterpret_tensor(buf949, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2095 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_175, sp_176], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf928, buf932, buf933, primals_215, primals_216, buf936, buf2095, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_216
        buf937 = reinterpret_tensor(buf927, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf927  # reuse
        # Source Nodes: [sp_177], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf936, buf915, buf937, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_178], Original ATen: [aten.convolution]
        buf938 = extern_kernels.convolution(buf937, buf42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf938, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf939 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_178], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf938, buf939, 832, 196, grid=grid(832, 196), stream=stream0)
        buf940 = buf931; del buf931  # reuse
        buf941 = buf930; del buf930  # reuse
        buf942 = buf929; del buf929  # reuse
        # Source Nodes: [sp_179], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf939, buf940, buf941, buf942, 1352, 121, grid=grid(1352), stream=stream0)
        buf943 = buf933; del buf933  # reuse
        buf944 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf946 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_179], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf940, buf941, buf942, primals_729, primals_730, buf943, buf944, buf946, primals_729, primals_730, 104, 13, grid=grid(104), stream=stream0)
        del primals_729
        del primals_730
        buf947 = reinterpret_tensor(buf949, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2094 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_179, sp_180], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf939, buf943, buf944, primals_218, primals_219, buf947, buf2094, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_219
        buf948 = reinterpret_tensor(buf949, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_52], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf915, buf948, 832, 196, grid=grid(832, 196), stream=stream0)
        buf950 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_52], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf949, buf950, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf925
        del buf936
        del buf947
        del buf948
        # Source Nodes: [out_108], Original ATen: [aten.convolution]
        buf951 = extern_kernels.convolution(buf950, primals_220, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf951, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf952 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_108], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf951, buf952, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf953 = buf900; del buf900  # reuse
        buf954 = buf899; del buf899  # reuse
        buf955 = buf898; del buf898  # reuse
        # Source Nodes: [out_109], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf952, buf953, buf954, buf955, 13312, 121, grid=grid(13312), stream=stream0)
        buf956 = buf902; del buf902  # reuse
        buf957 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf959 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_109], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf953, buf954, buf955, primals_732, primals_733, buf956, buf957, buf959, primals_732, primals_733, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_732
        del primals_733
        buf960 = reinterpret_tensor(buf951, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf951  # reuse
        # Source Nodes: [out_109, out_110, shortcut_17], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf952, buf956, buf957, primals_221, primals_222, buf905, buf960, 1605632, grid=grid(1605632), stream=stream0)
        del primals_222
        # Source Nodes: [out_112], Original ATen: [aten.convolution]
        buf961 = extern_kernels.convolution(buf960, primals_223, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf961, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf962 = reinterpret_tensor(buf949, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf949  # reuse
        # Source Nodes: [out_112], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf961, buf962, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf963 = buf910; del buf910  # reuse
        buf964 = buf909; del buf909  # reuse
        buf965 = buf908; del buf908  # reuse
        # Source Nodes: [out_113], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf962, buf963, buf964, buf965, 5408, 121, grid=grid(5408), stream=stream0)
        buf966 = buf912; del buf912  # reuse
        buf967 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf969 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_113], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf963, buf964, buf965, primals_735, primals_736, buf966, buf967, buf969, primals_735, primals_736, 416, 13, grid=grid(416), stream=stream0)
        del primals_735
        del primals_736
        buf970 = reinterpret_tensor(buf961, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf961  # reuse
        buf2093 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_113, out_114], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf962, buf966, buf967, primals_224, primals_225, buf970, buf2093, 652288, grid=grid(652288), stream=stream0)
        del primals_225
        # Source Nodes: [sp_183], Original ATen: [aten.convolution]
        buf971 = extern_kernels.convolution(reinterpret_tensor(buf970, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf971, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf972 = reinterpret_tensor(buf938, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf938  # reuse
        # Source Nodes: [sp_183], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf971, buf972, 832, 196, grid=grid(832, 196), stream=stream0)
        buf973 = buf942; del buf942  # reuse
        buf974 = buf941; del buf941  # reuse
        buf975 = buf940; del buf940  # reuse
        # Source Nodes: [sp_184], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf972, buf973, buf974, buf975, 1352, 121, grid=grid(1352), stream=stream0)
        buf976 = buf944; del buf944  # reuse
        buf977 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf979 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_184], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf973, buf974, buf975, primals_738, primals_739, buf976, buf977, buf979, primals_738, primals_739, 104, 13, grid=grid(104), stream=stream0)
        del primals_738
        del primals_739
        buf1004 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf980 = reinterpret_tensor(buf1004, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2092 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_184, sp_185], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf972, buf976, buf977, primals_227, primals_228, buf980, buf2092, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_228
        buf981 = reinterpret_tensor(buf971, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf971  # reuse
        # Source Nodes: [sp_186], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf980, buf970, buf981, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_187], Original ATen: [aten.convolution]
        buf982 = extern_kernels.convolution(buf981, buf44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf982, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf983 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_187], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf982, buf983, 832, 196, grid=grid(832, 196), stream=stream0)
        buf984 = buf975; del buf975  # reuse
        buf985 = buf974; del buf974  # reuse
        buf986 = buf973; del buf973  # reuse
        # Source Nodes: [sp_188], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf983, buf984, buf985, buf986, 1352, 121, grid=grid(1352), stream=stream0)
        buf987 = buf977; del buf977  # reuse
        buf988 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf990 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_188], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf984, buf985, buf986, primals_741, primals_742, buf987, buf988, buf990, primals_741, primals_742, 104, 13, grid=grid(104), stream=stream0)
        del primals_741
        del primals_742
        buf991 = reinterpret_tensor(buf1004, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2091 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_188, sp_189], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf983, buf987, buf988, primals_230, primals_231, buf991, buf2091, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_231
        buf992 = reinterpret_tensor(buf982, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf982  # reuse
        # Source Nodes: [sp_190], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf991, buf970, buf992, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_191], Original ATen: [aten.convolution]
        buf993 = extern_kernels.convolution(buf992, buf45, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf993, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf994 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_191], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf993, buf994, 832, 196, grid=grid(832, 196), stream=stream0)
        buf995 = buf986; del buf986  # reuse
        buf996 = buf985; del buf985  # reuse
        buf997 = buf984; del buf984  # reuse
        # Source Nodes: [sp_192], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf994, buf995, buf996, buf997, 1352, 121, grid=grid(1352), stream=stream0)
        buf998 = buf988; del buf988  # reuse
        buf999 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1001 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_192], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf995, buf996, buf997, primals_744, primals_745, buf998, buf999, buf1001, primals_744, primals_745, 104, 13, grid=grid(104), stream=stream0)
        del primals_744
        del primals_745
        buf1002 = reinterpret_tensor(buf1004, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2090 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_192, sp_193], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf994, buf998, buf999, primals_233, primals_234, buf1002, buf2090, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_234
        buf1003 = reinterpret_tensor(buf1004, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_51], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf970, buf1003, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1005 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_51], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf1004, buf1005, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf1002
        del buf1003
        del buf980
        del buf991
        # Source Nodes: [out_116], Original ATen: [aten.convolution]
        buf1006 = extern_kernels.convolution(buf1005, primals_235, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1006, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1007 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_116], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf1006, buf1007, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1008 = buf955; del buf955  # reuse
        buf1009 = buf954; del buf954  # reuse
        buf1010 = buf953; del buf953  # reuse
        # Source Nodes: [out_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf1007, buf1008, buf1009, buf1010, 13312, 121, grid=grid(13312), stream=stream0)
        buf1011 = buf957; del buf957  # reuse
        buf1012 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1014 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf1008, buf1009, buf1010, primals_747, primals_748, buf1011, buf1012, buf1014, primals_747, primals_748, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_747
        del primals_748
        buf1015 = reinterpret_tensor(buf1006, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1006  # reuse
        # Source Nodes: [out_117, out_118, shortcut_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf1007, buf1011, buf1012, primals_236, primals_237, buf960, buf1015, 1605632, grid=grid(1605632), stream=stream0)
        del primals_237
        # Source Nodes: [out_120], Original ATen: [aten.convolution]
        buf1016 = extern_kernels.convolution(buf1015, primals_238, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1016, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf1017 = reinterpret_tensor(buf1004, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1004  # reuse
        # Source Nodes: [out_120], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf1016, buf1017, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf1018 = buf965; del buf965  # reuse
        buf1019 = buf964; del buf964  # reuse
        buf1020 = buf963; del buf963  # reuse
        # Source Nodes: [out_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf1017, buf1018, buf1019, buf1020, 5408, 121, grid=grid(5408), stream=stream0)
        buf1021 = buf967; del buf967  # reuse
        buf1022 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf1024 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf1018, buf1019, buf1020, primals_750, primals_751, buf1021, buf1022, buf1024, primals_750, primals_751, 416, 13, grid=grid(416), stream=stream0)
        del primals_750
        del primals_751
        buf1025 = reinterpret_tensor(buf1016, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1016  # reuse
        buf2089 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_121, out_122], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf1017, buf1021, buf1022, primals_239, primals_240, buf1025, buf2089, 652288, grid=grid(652288), stream=stream0)
        del primals_240
        # Source Nodes: [sp_196], Original ATen: [aten.convolution]
        buf1026 = extern_kernels.convolution(reinterpret_tensor(buf1025, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1026, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1027 = reinterpret_tensor(buf993, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf993  # reuse
        # Source Nodes: [sp_196], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1026, buf1027, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1028 = buf997; del buf997  # reuse
        buf1029 = buf996; del buf996  # reuse
        buf1030 = buf995; del buf995  # reuse
        # Source Nodes: [sp_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1027, buf1028, buf1029, buf1030, 1352, 121, grid=grid(1352), stream=stream0)
        buf1031 = buf999; del buf999  # reuse
        buf1032 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1034 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1028, buf1029, buf1030, primals_753, primals_754, buf1031, buf1032, buf1034, primals_753, primals_754, 104, 13, grid=grid(104), stream=stream0)
        del primals_753
        del primals_754
        buf1059 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf1035 = reinterpret_tensor(buf1059, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2088 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_197, sp_198], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1027, buf1031, buf1032, primals_242, primals_243, buf1035, buf2088, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_243
        buf1036 = reinterpret_tensor(buf1026, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1026  # reuse
        # Source Nodes: [sp_199], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf1035, buf1025, buf1036, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_200], Original ATen: [aten.convolution]
        buf1037 = extern_kernels.convolution(buf1036, buf47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1037, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1038 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_200], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1037, buf1038, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1039 = buf1030; del buf1030  # reuse
        buf1040 = buf1029; del buf1029  # reuse
        buf1041 = buf1028; del buf1028  # reuse
        # Source Nodes: [sp_201], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1038, buf1039, buf1040, buf1041, 1352, 121, grid=grid(1352), stream=stream0)
        buf1042 = buf1032; del buf1032  # reuse
        buf1043 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1045 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_201], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1039, buf1040, buf1041, primals_756, primals_757, buf1042, buf1043, buf1045, primals_756, primals_757, 104, 13, grid=grid(104), stream=stream0)
        del primals_756
        del primals_757
        buf1046 = reinterpret_tensor(buf1059, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2087 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_201, sp_202], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1038, buf1042, buf1043, primals_245, primals_246, buf1046, buf2087, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_246
        buf1047 = reinterpret_tensor(buf1037, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1037  # reuse
        # Source Nodes: [sp_203], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf1046, buf1025, buf1047, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_204], Original ATen: [aten.convolution]
        buf1048 = extern_kernels.convolution(buf1047, buf48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1048, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1049 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_204], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1048, buf1049, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1050 = buf1041; del buf1041  # reuse
        buf1051 = buf1040; del buf1040  # reuse
        buf1052 = buf1039; del buf1039  # reuse
        # Source Nodes: [sp_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1049, buf1050, buf1051, buf1052, 1352, 121, grid=grid(1352), stream=stream0)
        buf1053 = buf1043; del buf1043  # reuse
        buf1054 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1056 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1050, buf1051, buf1052, primals_759, primals_760, buf1053, buf1054, buf1056, primals_759, primals_760, 104, 13, grid=grid(104), stream=stream0)
        del primals_759
        del primals_760
        buf1057 = reinterpret_tensor(buf1059, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2086 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_205, sp_206], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1049, buf1053, buf1054, primals_248, primals_249, buf1057, buf2086, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_249
        buf1058 = reinterpret_tensor(buf1059, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_50], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf1025, buf1058, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1060 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_50], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf1059, buf1060, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf1035
        del buf1046
        del buf1057
        del buf1058
        # Source Nodes: [out_124], Original ATen: [aten.convolution]
        buf1061 = extern_kernels.convolution(buf1060, primals_250, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1061, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1062 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_124], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf1061, buf1062, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1063 = buf1010; del buf1010  # reuse
        buf1064 = buf1009; del buf1009  # reuse
        buf1065 = buf1008; del buf1008  # reuse
        # Source Nodes: [out_125], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf1062, buf1063, buf1064, buf1065, 13312, 121, grid=grid(13312), stream=stream0)
        buf1066 = buf1012; del buf1012  # reuse
        buf1067 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1069 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_125], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf1063, buf1064, buf1065, primals_762, primals_763, buf1066, buf1067, buf1069, primals_762, primals_763, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_762
        del primals_763
        buf1070 = reinterpret_tensor(buf1061, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1061  # reuse
        # Source Nodes: [out_125, out_126, shortcut_19], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf1062, buf1066, buf1067, primals_251, primals_252, buf1015, buf1070, 1605632, grid=grid(1605632), stream=stream0)
        del primals_252
        # Source Nodes: [out_128], Original ATen: [aten.convolution]
        buf1071 = extern_kernels.convolution(buf1070, primals_253, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1071, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf1072 = reinterpret_tensor(buf1059, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1059  # reuse
        # Source Nodes: [out_128], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf1071, buf1072, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf1073 = buf1020; del buf1020  # reuse
        buf1074 = buf1019; del buf1019  # reuse
        buf1075 = buf1018; del buf1018  # reuse
        # Source Nodes: [out_129], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf1072, buf1073, buf1074, buf1075, 5408, 121, grid=grid(5408), stream=stream0)
        buf1076 = buf1022; del buf1022  # reuse
        buf1077 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf1079 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_129], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf1073, buf1074, buf1075, primals_765, primals_766, buf1076, buf1077, buf1079, primals_765, primals_766, 416, 13, grid=grid(416), stream=stream0)
        del primals_765
        del primals_766
        buf1080 = reinterpret_tensor(buf1071, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1071  # reuse
        buf2085 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_129, out_130], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf1072, buf1076, buf1077, primals_254, primals_255, buf1080, buf2085, 652288, grid=grid(652288), stream=stream0)
        del primals_255
        # Source Nodes: [sp_209], Original ATen: [aten.convolution]
        buf1081 = extern_kernels.convolution(reinterpret_tensor(buf1080, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1081, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1082 = reinterpret_tensor(buf1048, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1048  # reuse
        # Source Nodes: [sp_209], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1081, buf1082, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1083 = buf1052; del buf1052  # reuse
        buf1084 = buf1051; del buf1051  # reuse
        buf1085 = buf1050; del buf1050  # reuse
        # Source Nodes: [sp_210], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1082, buf1083, buf1084, buf1085, 1352, 121, grid=grid(1352), stream=stream0)
        buf1086 = buf1054; del buf1054  # reuse
        buf1087 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1089 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_210], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1083, buf1084, buf1085, primals_768, primals_769, buf1086, buf1087, buf1089, primals_768, primals_769, 104, 13, grid=grid(104), stream=stream0)
        del primals_768
        del primals_769
        buf1114 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf1090 = reinterpret_tensor(buf1114, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2084 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_210, sp_211], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1082, buf1086, buf1087, primals_257, primals_258, buf1090, buf2084, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_258
        buf1091 = reinterpret_tensor(buf1081, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1081  # reuse
        # Source Nodes: [sp_212], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf1090, buf1080, buf1091, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_213], Original ATen: [aten.convolution]
        buf1092 = extern_kernels.convolution(buf1091, buf50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1092, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1093 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_213], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1092, buf1093, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1094 = buf1085; del buf1085  # reuse
        buf1095 = buf1084; del buf1084  # reuse
        buf1096 = buf1083; del buf1083  # reuse
        # Source Nodes: [sp_214], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1093, buf1094, buf1095, buf1096, 1352, 121, grid=grid(1352), stream=stream0)
        buf1097 = buf1087; del buf1087  # reuse
        buf1098 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1100 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_214], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1094, buf1095, buf1096, primals_771, primals_772, buf1097, buf1098, buf1100, primals_771, primals_772, 104, 13, grid=grid(104), stream=stream0)
        del primals_771
        del primals_772
        buf1101 = reinterpret_tensor(buf1114, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2083 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_214, sp_215], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1093, buf1097, buf1098, primals_260, primals_261, buf1101, buf2083, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_261
        buf1102 = reinterpret_tensor(buf1092, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1092  # reuse
        # Source Nodes: [sp_216], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf1101, buf1080, buf1102, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_217], Original ATen: [aten.convolution]
        buf1103 = extern_kernels.convolution(buf1102, buf51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1103, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1104 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_217], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1103, buf1104, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1105 = buf1096; del buf1096  # reuse
        buf1106 = buf1095; del buf1095  # reuse
        buf1107 = buf1094; del buf1094  # reuse
        # Source Nodes: [sp_218], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1104, buf1105, buf1106, buf1107, 1352, 121, grid=grid(1352), stream=stream0)
        buf1108 = buf1098; del buf1098  # reuse
        buf1109 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1111 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_218], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1105, buf1106, buf1107, primals_774, primals_775, buf1108, buf1109, buf1111, primals_774, primals_775, 104, 13, grid=grid(104), stream=stream0)
        del primals_774
        del primals_775
        buf1112 = reinterpret_tensor(buf1114, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2082 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_218, sp_219], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1104, buf1108, buf1109, primals_263, primals_264, buf1112, buf2082, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_264
        buf1113 = reinterpret_tensor(buf1114, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_49], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf1080, buf1113, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1115 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_49], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf1114, buf1115, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf1090
        del buf1101
        del buf1112
        del buf1113
        # Source Nodes: [out_132], Original ATen: [aten.convolution]
        buf1116 = extern_kernels.convolution(buf1115, primals_265, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1116, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1117 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_132], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf1116, buf1117, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1118 = buf1065; del buf1065  # reuse
        buf1119 = buf1064; del buf1064  # reuse
        buf1120 = buf1063; del buf1063  # reuse
        # Source Nodes: [out_133], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf1117, buf1118, buf1119, buf1120, 13312, 121, grid=grid(13312), stream=stream0)
        buf1121 = buf1067; del buf1067  # reuse
        buf1122 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1124 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_133], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf1118, buf1119, buf1120, primals_777, primals_778, buf1121, buf1122, buf1124, primals_777, primals_778, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_777
        del primals_778
        buf1125 = reinterpret_tensor(buf1116, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1116  # reuse
        # Source Nodes: [out_133, out_134, shortcut_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf1117, buf1121, buf1122, primals_266, primals_267, buf1070, buf1125, 1605632, grid=grid(1605632), stream=stream0)
        del primals_267
        # Source Nodes: [out_136], Original ATen: [aten.convolution]
        buf1126 = extern_kernels.convolution(buf1125, primals_268, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1126, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf1127 = reinterpret_tensor(buf1114, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1114  # reuse
        # Source Nodes: [out_136], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf1126, buf1127, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf1128 = buf1075; del buf1075  # reuse
        buf1129 = buf1074; del buf1074  # reuse
        buf1130 = buf1073; del buf1073  # reuse
        # Source Nodes: [out_137], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf1127, buf1128, buf1129, buf1130, 5408, 121, grid=grid(5408), stream=stream0)
        buf1131 = buf1077; del buf1077  # reuse
        buf1132 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf1134 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_137], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf1128, buf1129, buf1130, primals_780, primals_781, buf1131, buf1132, buf1134, primals_780, primals_781, 416, 13, grid=grid(416), stream=stream0)
        del primals_780
        del primals_781
        buf1135 = reinterpret_tensor(buf1126, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1126  # reuse
        buf2081 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_137, out_138], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf1127, buf1131, buf1132, primals_269, primals_270, buf1135, buf2081, 652288, grid=grid(652288), stream=stream0)
        del primals_270
        # Source Nodes: [sp_222], Original ATen: [aten.convolution]
        buf1136 = extern_kernels.convolution(reinterpret_tensor(buf1135, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1136, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1137 = reinterpret_tensor(buf1103, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1103  # reuse
        # Source Nodes: [sp_222], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1136, buf1137, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1138 = buf1107; del buf1107  # reuse
        buf1139 = buf1106; del buf1106  # reuse
        buf1140 = buf1105; del buf1105  # reuse
        # Source Nodes: [sp_223], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1137, buf1138, buf1139, buf1140, 1352, 121, grid=grid(1352), stream=stream0)
        buf1141 = buf1109; del buf1109  # reuse
        buf1142 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1144 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_223], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1138, buf1139, buf1140, primals_783, primals_784, buf1141, buf1142, buf1144, primals_783, primals_784, 104, 13, grid=grid(104), stream=stream0)
        del primals_783
        del primals_784
        buf1169 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf1145 = reinterpret_tensor(buf1169, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2080 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_223, sp_224], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1137, buf1141, buf1142, primals_272, primals_273, buf1145, buf2080, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_273
        buf1146 = reinterpret_tensor(buf1136, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1136  # reuse
        # Source Nodes: [sp_225], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf1145, buf1135, buf1146, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_226], Original ATen: [aten.convolution]
        buf1147 = extern_kernels.convolution(buf1146, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1147, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1148 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_226], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1147, buf1148, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1149 = buf1140; del buf1140  # reuse
        buf1150 = buf1139; del buf1139  # reuse
        buf1151 = buf1138; del buf1138  # reuse
        # Source Nodes: [sp_227], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1148, buf1149, buf1150, buf1151, 1352, 121, grid=grid(1352), stream=stream0)
        buf1152 = buf1142; del buf1142  # reuse
        buf1153 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1155 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_227], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1149, buf1150, buf1151, primals_786, primals_787, buf1152, buf1153, buf1155, primals_786, primals_787, 104, 13, grid=grid(104), stream=stream0)
        del primals_786
        del primals_787
        buf1156 = reinterpret_tensor(buf1169, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2079 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_227, sp_228], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1148, buf1152, buf1153, primals_275, primals_276, buf1156, buf2079, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_276
        buf1157 = reinterpret_tensor(buf1147, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1147  # reuse
        # Source Nodes: [sp_229], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf1156, buf1135, buf1157, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_230], Original ATen: [aten.convolution]
        buf1158 = extern_kernels.convolution(buf1157, buf54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1158, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1159 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_230], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1158, buf1159, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1160 = buf1151; del buf1151  # reuse
        buf1161 = buf1150; del buf1150  # reuse
        buf1162 = buf1149; del buf1149  # reuse
        # Source Nodes: [sp_231], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1159, buf1160, buf1161, buf1162, 1352, 121, grid=grid(1352), stream=stream0)
        buf1163 = buf1153; del buf1153  # reuse
        buf1164 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1166 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_231], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1160, buf1161, buf1162, primals_789, primals_790, buf1163, buf1164, buf1166, primals_789, primals_790, 104, 13, grid=grid(104), stream=stream0)
        del primals_789
        del primals_790
        buf1167 = reinterpret_tensor(buf1169, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2078 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_231, sp_232], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1159, buf1163, buf1164, primals_278, primals_279, buf1167, buf2078, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_279
        buf1168 = reinterpret_tensor(buf1169, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_48], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf1135, buf1168, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1170 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_48], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf1169, buf1170, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf1145
        del buf1156
        del buf1167
        del buf1168
        # Source Nodes: [out_140], Original ATen: [aten.convolution]
        buf1171 = extern_kernels.convolution(buf1170, primals_280, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1171, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1172 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_140], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf1171, buf1172, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1173 = buf1120; del buf1120  # reuse
        buf1174 = buf1119; del buf1119  # reuse
        buf1175 = buf1118; del buf1118  # reuse
        # Source Nodes: [out_141], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf1172, buf1173, buf1174, buf1175, 13312, 121, grid=grid(13312), stream=stream0)
        buf1176 = buf1122; del buf1122  # reuse
        buf1177 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1179 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_141], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf1173, buf1174, buf1175, primals_792, primals_793, buf1176, buf1177, buf1179, primals_792, primals_793, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_792
        del primals_793
        buf1180 = reinterpret_tensor(buf1171, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1171  # reuse
        # Source Nodes: [out_141, out_142, shortcut_21], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf1172, buf1176, buf1177, primals_281, primals_282, buf1125, buf1180, 1605632, grid=grid(1605632), stream=stream0)
        del primals_282
        # Source Nodes: [out_144], Original ATen: [aten.convolution]
        buf1181 = extern_kernels.convolution(buf1180, primals_283, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1181, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf1182 = reinterpret_tensor(buf1169, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1169  # reuse
        # Source Nodes: [out_144], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf1181, buf1182, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf1183 = buf1130; del buf1130  # reuse
        buf1184 = buf1129; del buf1129  # reuse
        buf1185 = buf1128; del buf1128  # reuse
        # Source Nodes: [out_145], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf1182, buf1183, buf1184, buf1185, 5408, 121, grid=grid(5408), stream=stream0)
        buf1186 = buf1132; del buf1132  # reuse
        buf1187 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf1189 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_145], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf1183, buf1184, buf1185, primals_795, primals_796, buf1186, buf1187, buf1189, primals_795, primals_796, 416, 13, grid=grid(416), stream=stream0)
        del primals_795
        del primals_796
        buf1190 = reinterpret_tensor(buf1181, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1181  # reuse
        buf2077 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_145, out_146], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf1182, buf1186, buf1187, primals_284, primals_285, buf1190, buf2077, 652288, grid=grid(652288), stream=stream0)
        del primals_285
        # Source Nodes: [sp_235], Original ATen: [aten.convolution]
        buf1191 = extern_kernels.convolution(reinterpret_tensor(buf1190, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf55, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1191, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1192 = reinterpret_tensor(buf1158, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1158  # reuse
        # Source Nodes: [sp_235], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1191, buf1192, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1193 = buf1162; del buf1162  # reuse
        buf1194 = buf1161; del buf1161  # reuse
        buf1195 = buf1160; del buf1160  # reuse
        # Source Nodes: [sp_236], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1192, buf1193, buf1194, buf1195, 1352, 121, grid=grid(1352), stream=stream0)
        buf1196 = buf1164; del buf1164  # reuse
        buf1197 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1199 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_236], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1193, buf1194, buf1195, primals_798, primals_799, buf1196, buf1197, buf1199, primals_798, primals_799, 104, 13, grid=grid(104), stream=stream0)
        del primals_798
        del primals_799
        buf1224 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf1200 = reinterpret_tensor(buf1224, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2076 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_236, sp_237], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1192, buf1196, buf1197, primals_287, primals_288, buf1200, buf2076, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_288
        buf1201 = reinterpret_tensor(buf1191, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1191  # reuse
        # Source Nodes: [sp_238], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf1200, buf1190, buf1201, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_239], Original ATen: [aten.convolution]
        buf1202 = extern_kernels.convolution(buf1201, buf56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1202, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1203 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_239], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1202, buf1203, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1204 = buf1195; del buf1195  # reuse
        buf1205 = buf1194; del buf1194  # reuse
        buf1206 = buf1193; del buf1193  # reuse
        # Source Nodes: [sp_240], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1203, buf1204, buf1205, buf1206, 1352, 121, grid=grid(1352), stream=stream0)
        buf1207 = buf1197; del buf1197  # reuse
        buf1208 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1210 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_240], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1204, buf1205, buf1206, primals_801, primals_802, buf1207, buf1208, buf1210, primals_801, primals_802, 104, 13, grid=grid(104), stream=stream0)
        del primals_801
        del primals_802
        buf1211 = reinterpret_tensor(buf1224, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2075 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_240, sp_241], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1203, buf1207, buf1208, primals_290, primals_291, buf1211, buf2075, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_291
        buf1212 = reinterpret_tensor(buf1202, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1202  # reuse
        # Source Nodes: [sp_242], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf1211, buf1190, buf1212, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_243], Original ATen: [aten.convolution]
        buf1213 = extern_kernels.convolution(buf1212, buf57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1213, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1214 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_243], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1213, buf1214, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1215 = buf1206; del buf1206  # reuse
        buf1216 = buf1205; del buf1205  # reuse
        buf1217 = buf1204; del buf1204  # reuse
        # Source Nodes: [sp_244], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1214, buf1215, buf1216, buf1217, 1352, 121, grid=grid(1352), stream=stream0)
        buf1218 = buf1208; del buf1208  # reuse
        buf1219 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1221 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_244], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1215, buf1216, buf1217, primals_804, primals_805, buf1218, buf1219, buf1221, primals_804, primals_805, 104, 13, grid=grid(104), stream=stream0)
        del primals_804
        del primals_805
        buf1222 = reinterpret_tensor(buf1224, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2074 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_244, sp_245], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1214, buf1218, buf1219, primals_293, primals_294, buf1222, buf2074, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_294
        buf1223 = reinterpret_tensor(buf1224, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_47], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf1190, buf1223, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1225 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_47], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf1224, buf1225, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf1200
        del buf1211
        del buf1222
        del buf1223
        # Source Nodes: [out_148], Original ATen: [aten.convolution]
        buf1226 = extern_kernels.convolution(buf1225, primals_295, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1226, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1227 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_148], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf1226, buf1227, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1228 = buf1175; del buf1175  # reuse
        buf1229 = buf1174; del buf1174  # reuse
        buf1230 = buf1173; del buf1173  # reuse
        # Source Nodes: [out_149], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf1227, buf1228, buf1229, buf1230, 13312, 121, grid=grid(13312), stream=stream0)
        buf1231 = buf1177; del buf1177  # reuse
        buf1232 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1234 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_149], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf1228, buf1229, buf1230, primals_807, primals_808, buf1231, buf1232, buf1234, primals_807, primals_808, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_807
        del primals_808
        buf1235 = reinterpret_tensor(buf1226, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1226  # reuse
        # Source Nodes: [out_149, out_150, shortcut_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf1227, buf1231, buf1232, primals_296, primals_297, buf1180, buf1235, 1605632, grid=grid(1605632), stream=stream0)
        del primals_297
        # Source Nodes: [out_152], Original ATen: [aten.convolution]
        buf1236 = extern_kernels.convolution(buf1235, primals_298, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1236, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf1237 = reinterpret_tensor(buf1224, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1224  # reuse
        # Source Nodes: [out_152], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf1236, buf1237, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf1238 = buf1185; del buf1185  # reuse
        buf1239 = buf1184; del buf1184  # reuse
        buf1240 = buf1183; del buf1183  # reuse
        # Source Nodes: [out_153], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf1237, buf1238, buf1239, buf1240, 5408, 121, grid=grid(5408), stream=stream0)
        buf1241 = buf1187; del buf1187  # reuse
        buf1242 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf1244 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_153], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf1238, buf1239, buf1240, primals_810, primals_811, buf1241, buf1242, buf1244, primals_810, primals_811, 416, 13, grid=grid(416), stream=stream0)
        del primals_810
        del primals_811
        buf1245 = reinterpret_tensor(buf1236, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1236  # reuse
        buf2073 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_153, out_154], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf1237, buf1241, buf1242, primals_299, primals_300, buf1245, buf2073, 652288, grid=grid(652288), stream=stream0)
        del primals_300
        # Source Nodes: [sp_248], Original ATen: [aten.convolution]
        buf1246 = extern_kernels.convolution(reinterpret_tensor(buf1245, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1246, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1247 = reinterpret_tensor(buf1213, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1213  # reuse
        # Source Nodes: [sp_248], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1246, buf1247, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1248 = buf1217; del buf1217  # reuse
        buf1249 = buf1216; del buf1216  # reuse
        buf1250 = buf1215; del buf1215  # reuse
        # Source Nodes: [sp_249], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1247, buf1248, buf1249, buf1250, 1352, 121, grid=grid(1352), stream=stream0)
        buf1251 = buf1219; del buf1219  # reuse
        buf1252 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1254 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_249], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1248, buf1249, buf1250, primals_813, primals_814, buf1251, buf1252, buf1254, primals_813, primals_814, 104, 13, grid=grid(104), stream=stream0)
        del primals_813
        del primals_814
        buf1279 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf1255 = reinterpret_tensor(buf1279, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2072 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_249, sp_250], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1247, buf1251, buf1252, primals_302, primals_303, buf1255, buf2072, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_303
        buf1256 = reinterpret_tensor(buf1246, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1246  # reuse
        # Source Nodes: [sp_251], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf1255, buf1245, buf1256, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_252], Original ATen: [aten.convolution]
        buf1257 = extern_kernels.convolution(buf1256, buf59, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1257, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1258 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_252], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1257, buf1258, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1259 = buf1250; del buf1250  # reuse
        buf1260 = buf1249; del buf1249  # reuse
        buf1261 = buf1248; del buf1248  # reuse
        # Source Nodes: [sp_253], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1258, buf1259, buf1260, buf1261, 1352, 121, grid=grid(1352), stream=stream0)
        buf1262 = buf1252; del buf1252  # reuse
        buf1263 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1265 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_253], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1259, buf1260, buf1261, primals_816, primals_817, buf1262, buf1263, buf1265, primals_816, primals_817, 104, 13, grid=grid(104), stream=stream0)
        del primals_816
        del primals_817
        buf1266 = reinterpret_tensor(buf1279, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2071 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_253, sp_254], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1258, buf1262, buf1263, primals_305, primals_306, buf1266, buf2071, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_306
        buf1267 = reinterpret_tensor(buf1257, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1257  # reuse
        # Source Nodes: [sp_255], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf1266, buf1245, buf1267, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_256], Original ATen: [aten.convolution]
        buf1268 = extern_kernels.convolution(buf1267, buf60, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1268, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1269 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_256], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1268, buf1269, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1270 = buf1261; del buf1261  # reuse
        buf1271 = buf1260; del buf1260  # reuse
        buf1272 = buf1259; del buf1259  # reuse
        # Source Nodes: [sp_257], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1269, buf1270, buf1271, buf1272, 1352, 121, grid=grid(1352), stream=stream0)
        buf1273 = buf1263; del buf1263  # reuse
        buf1274 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1276 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_257], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1270, buf1271, buf1272, primals_819, primals_820, buf1273, buf1274, buf1276, primals_819, primals_820, 104, 13, grid=grid(104), stream=stream0)
        del primals_819
        del primals_820
        buf1277 = reinterpret_tensor(buf1279, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2070 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_257, sp_258], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1269, buf1273, buf1274, primals_308, primals_309, buf1277, buf2070, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_309
        buf1278 = reinterpret_tensor(buf1279, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_46], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf1245, buf1278, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1280 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_46], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf1279, buf1280, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf1255
        del buf1266
        del buf1277
        del buf1278
        # Source Nodes: [out_156], Original ATen: [aten.convolution]
        buf1281 = extern_kernels.convolution(buf1280, primals_310, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1281, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1282 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_156], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf1281, buf1282, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1283 = buf1230; del buf1230  # reuse
        buf1284 = buf1229; del buf1229  # reuse
        buf1285 = buf1228; del buf1228  # reuse
        # Source Nodes: [out_157], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf1282, buf1283, buf1284, buf1285, 13312, 121, grid=grid(13312), stream=stream0)
        buf1286 = buf1232; del buf1232  # reuse
        buf1287 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1289 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_157], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf1283, buf1284, buf1285, primals_822, primals_823, buf1286, buf1287, buf1289, primals_822, primals_823, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_822
        del primals_823
        buf1290 = reinterpret_tensor(buf1281, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1281  # reuse
        # Source Nodes: [out_157, out_158, shortcut_23], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf1282, buf1286, buf1287, primals_311, primals_312, buf1235, buf1290, 1605632, grid=grid(1605632), stream=stream0)
        del primals_312
        # Source Nodes: [out_160], Original ATen: [aten.convolution]
        buf1291 = extern_kernels.convolution(buf1290, primals_313, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1291, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf1292 = reinterpret_tensor(buf1279, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1279  # reuse
        # Source Nodes: [out_160], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf1291, buf1292, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf1293 = buf1240; del buf1240  # reuse
        buf1294 = buf1239; del buf1239  # reuse
        buf1295 = buf1238; del buf1238  # reuse
        # Source Nodes: [out_161], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf1292, buf1293, buf1294, buf1295, 5408, 121, grid=grid(5408), stream=stream0)
        buf1296 = buf1242; del buf1242  # reuse
        buf1297 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf1299 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_161], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf1293, buf1294, buf1295, primals_825, primals_826, buf1296, buf1297, buf1299, primals_825, primals_826, 416, 13, grid=grid(416), stream=stream0)
        del primals_825
        del primals_826
        buf1300 = reinterpret_tensor(buf1291, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1291  # reuse
        buf2069 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_161, out_162], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf1292, buf1296, buf1297, primals_314, primals_315, buf1300, buf2069, 652288, grid=grid(652288), stream=stream0)
        del primals_315
        # Source Nodes: [sp_261], Original ATen: [aten.convolution]
        buf1301 = extern_kernels.convolution(reinterpret_tensor(buf1300, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1301, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1302 = reinterpret_tensor(buf1268, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1268  # reuse
        # Source Nodes: [sp_261], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1301, buf1302, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1303 = buf1272; del buf1272  # reuse
        buf1304 = buf1271; del buf1271  # reuse
        buf1305 = buf1270; del buf1270  # reuse
        # Source Nodes: [sp_262], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1302, buf1303, buf1304, buf1305, 1352, 121, grid=grid(1352), stream=stream0)
        buf1306 = buf1274; del buf1274  # reuse
        buf1307 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1309 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_262], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1303, buf1304, buf1305, primals_828, primals_829, buf1306, buf1307, buf1309, primals_828, primals_829, 104, 13, grid=grid(104), stream=stream0)
        del primals_828
        del primals_829
        buf1334 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf1310 = reinterpret_tensor(buf1334, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2068 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_262, sp_263], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1302, buf1306, buf1307, primals_317, primals_318, buf1310, buf2068, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_318
        buf1311 = reinterpret_tensor(buf1301, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1301  # reuse
        # Source Nodes: [sp_264], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf1310, buf1300, buf1311, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_265], Original ATen: [aten.convolution]
        buf1312 = extern_kernels.convolution(buf1311, buf62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1312, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1313 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_265], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1312, buf1313, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1314 = buf1305; del buf1305  # reuse
        buf1315 = buf1304; del buf1304  # reuse
        buf1316 = buf1303; del buf1303  # reuse
        # Source Nodes: [sp_266], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1313, buf1314, buf1315, buf1316, 1352, 121, grid=grid(1352), stream=stream0)
        buf1317 = buf1307; del buf1307  # reuse
        buf1318 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1320 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_266], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1314, buf1315, buf1316, primals_831, primals_832, buf1317, buf1318, buf1320, primals_831, primals_832, 104, 13, grid=grid(104), stream=stream0)
        del primals_831
        del primals_832
        buf1321 = reinterpret_tensor(buf1334, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2067 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_266, sp_267], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1313, buf1317, buf1318, primals_320, primals_321, buf1321, buf2067, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_321
        buf1322 = reinterpret_tensor(buf1312, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1312  # reuse
        # Source Nodes: [sp_268], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf1321, buf1300, buf1322, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_269], Original ATen: [aten.convolution]
        buf1323 = extern_kernels.convolution(buf1322, buf63, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1323, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1324 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_269], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1323, buf1324, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1325 = buf1316; del buf1316  # reuse
        buf1326 = buf1315; del buf1315  # reuse
        buf1327 = buf1314; del buf1314  # reuse
        # Source Nodes: [sp_270], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1324, buf1325, buf1326, buf1327, 1352, 121, grid=grid(1352), stream=stream0)
        buf1328 = buf1318; del buf1318  # reuse
        buf1329 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1331 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_270], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1325, buf1326, buf1327, primals_834, primals_835, buf1328, buf1329, buf1331, primals_834, primals_835, 104, 13, grid=grid(104), stream=stream0)
        del primals_834
        del primals_835
        buf1332 = reinterpret_tensor(buf1334, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2066 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_270, sp_271], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1324, buf1328, buf1329, primals_323, primals_324, buf1332, buf2066, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_324
        buf1333 = reinterpret_tensor(buf1334, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_45], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf1300, buf1333, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1335 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_45], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf1334, buf1335, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf1310
        del buf1321
        del buf1332
        del buf1333
        # Source Nodes: [out_164], Original ATen: [aten.convolution]
        buf1336 = extern_kernels.convolution(buf1335, primals_325, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1336, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1337 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_164], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf1336, buf1337, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1338 = buf1285; del buf1285  # reuse
        buf1339 = buf1284; del buf1284  # reuse
        buf1340 = buf1283; del buf1283  # reuse
        # Source Nodes: [out_165], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf1337, buf1338, buf1339, buf1340, 13312, 121, grid=grid(13312), stream=stream0)
        buf1341 = buf1287; del buf1287  # reuse
        buf1342 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1344 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_165], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf1338, buf1339, buf1340, primals_837, primals_838, buf1341, buf1342, buf1344, primals_837, primals_838, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_837
        del primals_838
        buf1345 = reinterpret_tensor(buf1336, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1336  # reuse
        # Source Nodes: [out_165, out_166, shortcut_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf1337, buf1341, buf1342, primals_326, primals_327, buf1290, buf1345, 1605632, grid=grid(1605632), stream=stream0)
        del primals_327
        # Source Nodes: [out_168], Original ATen: [aten.convolution]
        buf1346 = extern_kernels.convolution(buf1345, primals_328, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1346, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf1347 = reinterpret_tensor(buf1334, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1334  # reuse
        # Source Nodes: [out_168], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf1346, buf1347, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf1348 = buf1295; del buf1295  # reuse
        buf1349 = buf1294; del buf1294  # reuse
        buf1350 = buf1293; del buf1293  # reuse
        # Source Nodes: [out_169], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf1347, buf1348, buf1349, buf1350, 5408, 121, grid=grid(5408), stream=stream0)
        buf1351 = buf1297; del buf1297  # reuse
        buf1352 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf1354 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_169], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf1348, buf1349, buf1350, primals_840, primals_841, buf1351, buf1352, buf1354, primals_840, primals_841, 416, 13, grid=grid(416), stream=stream0)
        del primals_840
        del primals_841
        buf1355 = reinterpret_tensor(buf1346, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1346  # reuse
        buf2065 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_169, out_170], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf1347, buf1351, buf1352, primals_329, primals_330, buf1355, buf2065, 652288, grid=grid(652288), stream=stream0)
        del primals_330
        # Source Nodes: [sp_274], Original ATen: [aten.convolution]
        buf1356 = extern_kernels.convolution(reinterpret_tensor(buf1355, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf64, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1356, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1357 = reinterpret_tensor(buf1323, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1323  # reuse
        # Source Nodes: [sp_274], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1356, buf1357, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1358 = buf1327; del buf1327  # reuse
        buf1359 = buf1326; del buf1326  # reuse
        buf1360 = buf1325; del buf1325  # reuse
        # Source Nodes: [sp_275], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1357, buf1358, buf1359, buf1360, 1352, 121, grid=grid(1352), stream=stream0)
        buf1361 = buf1329; del buf1329  # reuse
        buf1362 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1364 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_275], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1358, buf1359, buf1360, primals_843, primals_844, buf1361, buf1362, buf1364, primals_843, primals_844, 104, 13, grid=grid(104), stream=stream0)
        del primals_843
        del primals_844
        buf1389 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf1365 = reinterpret_tensor(buf1389, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2064 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_275, sp_276], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1357, buf1361, buf1362, primals_332, primals_333, buf1365, buf2064, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_333
        buf1366 = reinterpret_tensor(buf1356, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1356  # reuse
        # Source Nodes: [sp_277], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf1365, buf1355, buf1366, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_278], Original ATen: [aten.convolution]
        buf1367 = extern_kernels.convolution(buf1366, buf65, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1367, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1368 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_278], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1367, buf1368, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1369 = buf1360; del buf1360  # reuse
        buf1370 = buf1359; del buf1359  # reuse
        buf1371 = buf1358; del buf1358  # reuse
        # Source Nodes: [sp_279], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1368, buf1369, buf1370, buf1371, 1352, 121, grid=grid(1352), stream=stream0)
        buf1372 = buf1362; del buf1362  # reuse
        buf1373 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1375 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_279], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1369, buf1370, buf1371, primals_846, primals_847, buf1372, buf1373, buf1375, primals_846, primals_847, 104, 13, grid=grid(104), stream=stream0)
        del primals_846
        del primals_847
        buf1376 = reinterpret_tensor(buf1389, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2063 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_279, sp_280], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1368, buf1372, buf1373, primals_335, primals_336, buf1376, buf2063, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_336
        buf1377 = reinterpret_tensor(buf1367, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1367  # reuse
        # Source Nodes: [sp_281], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf1376, buf1355, buf1377, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_282], Original ATen: [aten.convolution]
        buf1378 = extern_kernels.convolution(buf1377, buf66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1378, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1379 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_282], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1378, buf1379, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1380 = buf1371; del buf1371  # reuse
        buf1381 = buf1370; del buf1370  # reuse
        buf1382 = buf1369; del buf1369  # reuse
        # Source Nodes: [sp_283], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1379, buf1380, buf1381, buf1382, 1352, 121, grid=grid(1352), stream=stream0)
        buf1383 = buf1373; del buf1373  # reuse
        buf1384 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1386 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_283], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1380, buf1381, buf1382, primals_849, primals_850, buf1383, buf1384, buf1386, primals_849, primals_850, 104, 13, grid=grid(104), stream=stream0)
        del primals_849
        del primals_850
        buf1387 = reinterpret_tensor(buf1389, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2062 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_283, sp_284], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1379, buf1383, buf1384, primals_338, primals_339, buf1387, buf2062, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_339
        buf1388 = reinterpret_tensor(buf1389, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_44], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf1355, buf1388, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1390 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_44], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf1389, buf1390, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf1365
        del buf1376
        del buf1387
        del buf1388
        # Source Nodes: [out_172], Original ATen: [aten.convolution]
        buf1391 = extern_kernels.convolution(buf1390, primals_340, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1391, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1392 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_172], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf1391, buf1392, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1393 = buf1340; del buf1340  # reuse
        buf1394 = buf1339; del buf1339  # reuse
        buf1395 = buf1338; del buf1338  # reuse
        # Source Nodes: [out_173], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf1392, buf1393, buf1394, buf1395, 13312, 121, grid=grid(13312), stream=stream0)
        buf1396 = buf1342; del buf1342  # reuse
        buf1397 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1399 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_173], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf1393, buf1394, buf1395, primals_852, primals_853, buf1396, buf1397, buf1399, primals_852, primals_853, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_852
        del primals_853
        buf1400 = reinterpret_tensor(buf1391, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1391  # reuse
        # Source Nodes: [out_173, out_174, shortcut_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf1392, buf1396, buf1397, primals_341, primals_342, buf1345, buf1400, 1605632, grid=grid(1605632), stream=stream0)
        del primals_342
        # Source Nodes: [out_176], Original ATen: [aten.convolution]
        buf1401 = extern_kernels.convolution(buf1400, primals_343, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1401, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf1402 = reinterpret_tensor(buf1389, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1389  # reuse
        # Source Nodes: [out_176], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf1401, buf1402, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf1403 = buf1350; del buf1350  # reuse
        buf1404 = buf1349; del buf1349  # reuse
        buf1405 = buf1348; del buf1348  # reuse
        # Source Nodes: [out_177], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf1402, buf1403, buf1404, buf1405, 5408, 121, grid=grid(5408), stream=stream0)
        buf1406 = buf1352; del buf1352  # reuse
        buf1407 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf1409 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_177], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf1403, buf1404, buf1405, primals_855, primals_856, buf1406, buf1407, buf1409, primals_855, primals_856, 416, 13, grid=grid(416), stream=stream0)
        del primals_855
        del primals_856
        buf1410 = reinterpret_tensor(buf1401, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1401  # reuse
        buf2061 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_177, out_178], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf1402, buf1406, buf1407, primals_344, primals_345, buf1410, buf2061, 652288, grid=grid(652288), stream=stream0)
        del primals_345
        # Source Nodes: [sp_287], Original ATen: [aten.convolution]
        buf1411 = extern_kernels.convolution(reinterpret_tensor(buf1410, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1411, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1412 = reinterpret_tensor(buf1378, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1378  # reuse
        # Source Nodes: [sp_287], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1411, buf1412, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1413 = buf1382; del buf1382  # reuse
        buf1414 = buf1381; del buf1381  # reuse
        buf1415 = buf1380; del buf1380  # reuse
        # Source Nodes: [sp_288], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1412, buf1413, buf1414, buf1415, 1352, 121, grid=grid(1352), stream=stream0)
        buf1416 = buf1384; del buf1384  # reuse
        buf1417 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1419 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_288], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1413, buf1414, buf1415, primals_858, primals_859, buf1416, buf1417, buf1419, primals_858, primals_859, 104, 13, grid=grid(104), stream=stream0)
        del primals_858
        del primals_859
        buf1444 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf1420 = reinterpret_tensor(buf1444, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2060 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_288, sp_289], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1412, buf1416, buf1417, primals_347, primals_348, buf1420, buf2060, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_348
        buf1421 = reinterpret_tensor(buf1411, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1411  # reuse
        # Source Nodes: [sp_290], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf1420, buf1410, buf1421, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_291], Original ATen: [aten.convolution]
        buf1422 = extern_kernels.convolution(buf1421, buf68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1422, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1423 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_291], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1422, buf1423, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1424 = buf1415; del buf1415  # reuse
        buf1425 = buf1414; del buf1414  # reuse
        buf1426 = buf1413; del buf1413  # reuse
        # Source Nodes: [sp_292], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1423, buf1424, buf1425, buf1426, 1352, 121, grid=grid(1352), stream=stream0)
        buf1427 = buf1417; del buf1417  # reuse
        buf1428 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1430 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_292], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1424, buf1425, buf1426, primals_861, primals_862, buf1427, buf1428, buf1430, primals_861, primals_862, 104, 13, grid=grid(104), stream=stream0)
        del primals_861
        del primals_862
        buf1431 = reinterpret_tensor(buf1444, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2059 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_292, sp_293], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1423, buf1427, buf1428, primals_350, primals_351, buf1431, buf2059, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_351
        buf1432 = reinterpret_tensor(buf1422, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1422  # reuse
        # Source Nodes: [sp_294], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf1431, buf1410, buf1432, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_295], Original ATen: [aten.convolution]
        buf1433 = extern_kernels.convolution(buf1432, buf69, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1433, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1434 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_295], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1433, buf1434, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1435 = buf1426; del buf1426  # reuse
        buf1436 = buf1425; del buf1425  # reuse
        buf1437 = buf1424; del buf1424  # reuse
        # Source Nodes: [sp_296], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1434, buf1435, buf1436, buf1437, 1352, 121, grid=grid(1352), stream=stream0)
        buf1438 = buf1428; del buf1428  # reuse
        buf1439 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1441 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_296], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1435, buf1436, buf1437, primals_864, primals_865, buf1438, buf1439, buf1441, primals_864, primals_865, 104, 13, grid=grid(104), stream=stream0)
        del primals_864
        del primals_865
        buf1442 = reinterpret_tensor(buf1444, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2058 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_296, sp_297], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1434, buf1438, buf1439, primals_353, primals_354, buf1442, buf2058, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_354
        buf1443 = reinterpret_tensor(buf1444, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_43], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf1410, buf1443, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1445 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_43], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf1444, buf1445, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf1420
        del buf1431
        del buf1442
        del buf1443
        # Source Nodes: [out_180], Original ATen: [aten.convolution]
        buf1446 = extern_kernels.convolution(buf1445, primals_355, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1446, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1447 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_180], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf1446, buf1447, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1448 = buf1395; del buf1395  # reuse
        buf1449 = buf1394; del buf1394  # reuse
        buf1450 = buf1393; del buf1393  # reuse
        # Source Nodes: [out_181], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf1447, buf1448, buf1449, buf1450, 13312, 121, grid=grid(13312), stream=stream0)
        buf1451 = buf1397; del buf1397  # reuse
        buf1452 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1454 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_181], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf1448, buf1449, buf1450, primals_867, primals_868, buf1451, buf1452, buf1454, primals_867, primals_868, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_867
        del primals_868
        buf1455 = reinterpret_tensor(buf1446, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1446  # reuse
        # Source Nodes: [out_181, out_182, shortcut_26], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf1447, buf1451, buf1452, primals_356, primals_357, buf1400, buf1455, 1605632, grid=grid(1605632), stream=stream0)
        del primals_357
        # Source Nodes: [out_184], Original ATen: [aten.convolution]
        buf1456 = extern_kernels.convolution(buf1455, primals_358, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1456, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf1457 = reinterpret_tensor(buf1444, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1444  # reuse
        # Source Nodes: [out_184], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf1456, buf1457, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf1458 = buf1405; del buf1405  # reuse
        buf1459 = buf1404; del buf1404  # reuse
        buf1460 = buf1403; del buf1403  # reuse
        # Source Nodes: [out_185], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf1457, buf1458, buf1459, buf1460, 5408, 121, grid=grid(5408), stream=stream0)
        buf1461 = buf1407; del buf1407  # reuse
        buf1462 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf1464 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_185], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf1458, buf1459, buf1460, primals_870, primals_871, buf1461, buf1462, buf1464, primals_870, primals_871, 416, 13, grid=grid(416), stream=stream0)
        del primals_870
        del primals_871
        buf1465 = reinterpret_tensor(buf1456, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1456  # reuse
        buf2057 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_185, out_186], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf1457, buf1461, buf1462, primals_359, primals_360, buf1465, buf2057, 652288, grid=grid(652288), stream=stream0)
        del primals_360
        # Source Nodes: [sp_300], Original ATen: [aten.convolution]
        buf1466 = extern_kernels.convolution(reinterpret_tensor(buf1465, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf70, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1466, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1467 = reinterpret_tensor(buf1433, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1433  # reuse
        # Source Nodes: [sp_300], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1466, buf1467, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1468 = buf1437; del buf1437  # reuse
        buf1469 = buf1436; del buf1436  # reuse
        buf1470 = buf1435; del buf1435  # reuse
        # Source Nodes: [sp_301], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1467, buf1468, buf1469, buf1470, 1352, 121, grid=grid(1352), stream=stream0)
        buf1471 = buf1439; del buf1439  # reuse
        buf1472 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1474 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_301], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1468, buf1469, buf1470, primals_873, primals_874, buf1471, buf1472, buf1474, primals_873, primals_874, 104, 13, grid=grid(104), stream=stream0)
        del primals_873
        del primals_874
        buf1499 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf1475 = reinterpret_tensor(buf1499, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2056 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_301, sp_302], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1467, buf1471, buf1472, primals_362, primals_363, buf1475, buf2056, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_363
        buf1476 = reinterpret_tensor(buf1466, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1466  # reuse
        # Source Nodes: [sp_303], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf1475, buf1465, buf1476, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_304], Original ATen: [aten.convolution]
        buf1477 = extern_kernels.convolution(buf1476, buf71, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1477, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1478 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_304], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1477, buf1478, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1479 = buf1470; del buf1470  # reuse
        buf1480 = buf1469; del buf1469  # reuse
        buf1481 = buf1468; del buf1468  # reuse
        # Source Nodes: [sp_305], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1478, buf1479, buf1480, buf1481, 1352, 121, grid=grid(1352), stream=stream0)
        buf1482 = buf1472; del buf1472  # reuse
        buf1483 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1485 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_305], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1479, buf1480, buf1481, primals_876, primals_877, buf1482, buf1483, buf1485, primals_876, primals_877, 104, 13, grid=grid(104), stream=stream0)
        del primals_876
        del primals_877
        buf1486 = reinterpret_tensor(buf1499, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2055 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_305, sp_306], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1478, buf1482, buf1483, primals_365, primals_366, buf1486, buf2055, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_366
        buf1487 = reinterpret_tensor(buf1477, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1477  # reuse
        # Source Nodes: [sp_307], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf1486, buf1465, buf1487, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_308], Original ATen: [aten.convolution]
        buf1488 = extern_kernels.convolution(buf1487, buf72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1488, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1489 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_308], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1488, buf1489, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1490 = buf1481; del buf1481  # reuse
        buf1491 = buf1480; del buf1480  # reuse
        buf1492 = buf1479; del buf1479  # reuse
        # Source Nodes: [sp_309], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1489, buf1490, buf1491, buf1492, 1352, 121, grid=grid(1352), stream=stream0)
        buf1493 = buf1483; del buf1483  # reuse
        buf1494 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1496 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_309], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1490, buf1491, buf1492, primals_879, primals_880, buf1493, buf1494, buf1496, primals_879, primals_880, 104, 13, grid=grid(104), stream=stream0)
        del primals_879
        del primals_880
        buf1497 = reinterpret_tensor(buf1499, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2054 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_309, sp_310], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1489, buf1493, buf1494, primals_368, primals_369, buf1497, buf2054, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_369
        buf1498 = reinterpret_tensor(buf1499, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_42], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf1465, buf1498, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1500 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_42], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf1499, buf1500, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf1475
        del buf1486
        del buf1497
        del buf1498
        # Source Nodes: [out_188], Original ATen: [aten.convolution]
        buf1501 = extern_kernels.convolution(buf1500, primals_370, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1501, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1502 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_188], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf1501, buf1502, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1503 = buf1450; del buf1450  # reuse
        buf1504 = buf1449; del buf1449  # reuse
        buf1505 = buf1448; del buf1448  # reuse
        # Source Nodes: [out_189], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf1502, buf1503, buf1504, buf1505, 13312, 121, grid=grid(13312), stream=stream0)
        buf1506 = buf1452; del buf1452  # reuse
        buf1507 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1509 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_189], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf1503, buf1504, buf1505, primals_882, primals_883, buf1506, buf1507, buf1509, primals_882, primals_883, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_882
        del primals_883
        buf1510 = reinterpret_tensor(buf1501, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1501  # reuse
        # Source Nodes: [out_189, out_190, shortcut_27], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf1502, buf1506, buf1507, primals_371, primals_372, buf1455, buf1510, 1605632, grid=grid(1605632), stream=stream0)
        del primals_372
        # Source Nodes: [out_192], Original ATen: [aten.convolution]
        buf1511 = extern_kernels.convolution(buf1510, primals_373, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1511, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf1512 = reinterpret_tensor(buf1499, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1499  # reuse
        # Source Nodes: [out_192], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf1511, buf1512, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf1513 = buf1460; del buf1460  # reuse
        buf1514 = buf1459; del buf1459  # reuse
        buf1515 = buf1458; del buf1458  # reuse
        # Source Nodes: [out_193], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf1512, buf1513, buf1514, buf1515, 5408, 121, grid=grid(5408), stream=stream0)
        buf1516 = buf1462; del buf1462  # reuse
        buf1517 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf1519 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_193], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf1513, buf1514, buf1515, primals_885, primals_886, buf1516, buf1517, buf1519, primals_885, primals_886, 416, 13, grid=grid(416), stream=stream0)
        del primals_885
        del primals_886
        buf1520 = reinterpret_tensor(buf1511, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1511  # reuse
        buf2053 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_193, out_194], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf1512, buf1516, buf1517, primals_374, primals_375, buf1520, buf2053, 652288, grid=grid(652288), stream=stream0)
        del primals_375
        # Source Nodes: [sp_313], Original ATen: [aten.convolution]
        buf1521 = extern_kernels.convolution(reinterpret_tensor(buf1520, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1521, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1522 = reinterpret_tensor(buf1488, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1488  # reuse
        # Source Nodes: [sp_313], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1521, buf1522, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1523 = buf1492; del buf1492  # reuse
        buf1524 = buf1491; del buf1491  # reuse
        buf1525 = buf1490; del buf1490  # reuse
        # Source Nodes: [sp_314], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1522, buf1523, buf1524, buf1525, 1352, 121, grid=grid(1352), stream=stream0)
        buf1526 = buf1494; del buf1494  # reuse
        buf1527 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1529 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_314], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1523, buf1524, buf1525, primals_888, primals_889, buf1526, buf1527, buf1529, primals_888, primals_889, 104, 13, grid=grid(104), stream=stream0)
        del primals_888
        del primals_889
        buf1554 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf1530 = reinterpret_tensor(buf1554, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2052 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_314, sp_315], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1522, buf1526, buf1527, primals_377, primals_378, buf1530, buf2052, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_378
        buf1531 = reinterpret_tensor(buf1521, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1521  # reuse
        # Source Nodes: [sp_316], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf1530, buf1520, buf1531, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_317], Original ATen: [aten.convolution]
        buf1532 = extern_kernels.convolution(buf1531, buf74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1532, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1533 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_317], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1532, buf1533, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1534 = buf1525; del buf1525  # reuse
        buf1535 = buf1524; del buf1524  # reuse
        buf1536 = buf1523; del buf1523  # reuse
        # Source Nodes: [sp_318], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1533, buf1534, buf1535, buf1536, 1352, 121, grid=grid(1352), stream=stream0)
        buf1537 = buf1527; del buf1527  # reuse
        buf1538 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1540 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_318], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1534, buf1535, buf1536, primals_891, primals_892, buf1537, buf1538, buf1540, primals_891, primals_892, 104, 13, grid=grid(104), stream=stream0)
        del primals_891
        del primals_892
        buf1541 = reinterpret_tensor(buf1554, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2051 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_318, sp_319], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1533, buf1537, buf1538, primals_380, primals_381, buf1541, buf2051, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_381
        buf1542 = reinterpret_tensor(buf1532, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1532  # reuse
        # Source Nodes: [sp_320], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf1541, buf1520, buf1542, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_321], Original ATen: [aten.convolution]
        buf1543 = extern_kernels.convolution(buf1542, buf75, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1543, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1544 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_321], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1543, buf1544, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1545 = buf1536; del buf1536  # reuse
        buf1546 = buf1535; del buf1535  # reuse
        buf1547 = buf1534; del buf1534  # reuse
        # Source Nodes: [sp_322], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1544, buf1545, buf1546, buf1547, 1352, 121, grid=grid(1352), stream=stream0)
        buf1548 = buf1538; del buf1538  # reuse
        buf1549 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1551 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_322], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1545, buf1546, buf1547, primals_894, primals_895, buf1548, buf1549, buf1551, primals_894, primals_895, 104, 13, grid=grid(104), stream=stream0)
        del primals_894
        del primals_895
        buf1552 = reinterpret_tensor(buf1554, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2050 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_322, sp_323], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1544, buf1548, buf1549, primals_383, primals_384, buf1552, buf2050, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_384
        buf1553 = reinterpret_tensor(buf1554, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_41], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf1520, buf1553, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1555 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_41], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf1554, buf1555, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf1530
        del buf1541
        del buf1552
        del buf1553
        # Source Nodes: [out_196], Original ATen: [aten.convolution]
        buf1556 = extern_kernels.convolution(buf1555, primals_385, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1556, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1557 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_196], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf1556, buf1557, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1558 = buf1505; del buf1505  # reuse
        buf1559 = buf1504; del buf1504  # reuse
        buf1560 = buf1503; del buf1503  # reuse
        # Source Nodes: [out_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf1557, buf1558, buf1559, buf1560, 13312, 121, grid=grid(13312), stream=stream0)
        buf1561 = buf1507; del buf1507  # reuse
        buf1562 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1564 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf1558, buf1559, buf1560, primals_897, primals_898, buf1561, buf1562, buf1564, primals_897, primals_898, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_897
        del primals_898
        buf1565 = reinterpret_tensor(buf1556, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1556  # reuse
        # Source Nodes: [out_197, out_198, shortcut_28], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf1557, buf1561, buf1562, primals_386, primals_387, buf1510, buf1565, 1605632, grid=grid(1605632), stream=stream0)
        del primals_387
        # Source Nodes: [out_200], Original ATen: [aten.convolution]
        buf1566 = extern_kernels.convolution(buf1565, primals_388, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1566, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf1567 = reinterpret_tensor(buf1554, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1554  # reuse
        # Source Nodes: [out_200], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf1566, buf1567, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf1568 = buf1515; del buf1515  # reuse
        buf1569 = buf1514; del buf1514  # reuse
        buf1570 = buf1513; del buf1513  # reuse
        # Source Nodes: [out_201], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf1567, buf1568, buf1569, buf1570, 5408, 121, grid=grid(5408), stream=stream0)
        buf1571 = buf1517; del buf1517  # reuse
        buf1572 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf1574 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_201], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf1568, buf1569, buf1570, primals_900, primals_901, buf1571, buf1572, buf1574, primals_900, primals_901, 416, 13, grid=grid(416), stream=stream0)
        del primals_900
        del primals_901
        buf1575 = reinterpret_tensor(buf1566, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1566  # reuse
        buf2049 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_201, out_202], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf1567, buf1571, buf1572, primals_389, primals_390, buf1575, buf2049, 652288, grid=grid(652288), stream=stream0)
        del primals_390
        # Source Nodes: [sp_326], Original ATen: [aten.convolution]
        buf1576 = extern_kernels.convolution(reinterpret_tensor(buf1575, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1576, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1577 = reinterpret_tensor(buf1543, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1543  # reuse
        # Source Nodes: [sp_326], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1576, buf1577, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1578 = buf1547; del buf1547  # reuse
        buf1579 = buf1546; del buf1546  # reuse
        buf1580 = buf1545; del buf1545  # reuse
        # Source Nodes: [sp_327], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1577, buf1578, buf1579, buf1580, 1352, 121, grid=grid(1352), stream=stream0)
        buf1581 = buf1549; del buf1549  # reuse
        buf1582 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1584 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_327], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1578, buf1579, buf1580, primals_903, primals_904, buf1581, buf1582, buf1584, primals_903, primals_904, 104, 13, grid=grid(104), stream=stream0)
        del primals_903
        del primals_904
        buf1609 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf1585 = reinterpret_tensor(buf1609, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2048 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_327, sp_328], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1577, buf1581, buf1582, primals_392, primals_393, buf1585, buf2048, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_393
        buf1586 = reinterpret_tensor(buf1576, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1576  # reuse
        # Source Nodes: [sp_329], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf1585, buf1575, buf1586, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_330], Original ATen: [aten.convolution]
        buf1587 = extern_kernels.convolution(buf1586, buf77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1587, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1588 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_330], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1587, buf1588, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1589 = buf1580; del buf1580  # reuse
        buf1590 = buf1579; del buf1579  # reuse
        buf1591 = buf1578; del buf1578  # reuse
        # Source Nodes: [sp_331], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1588, buf1589, buf1590, buf1591, 1352, 121, grid=grid(1352), stream=stream0)
        buf1592 = buf1582; del buf1582  # reuse
        buf1593 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1595 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_331], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1589, buf1590, buf1591, primals_906, primals_907, buf1592, buf1593, buf1595, primals_906, primals_907, 104, 13, grid=grid(104), stream=stream0)
        del primals_906
        del primals_907
        buf1596 = reinterpret_tensor(buf1609, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2047 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_331, sp_332], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1588, buf1592, buf1593, primals_395, primals_396, buf1596, buf2047, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_396
        buf1597 = reinterpret_tensor(buf1587, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1587  # reuse
        # Source Nodes: [sp_333], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf1596, buf1575, buf1597, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_334], Original ATen: [aten.convolution]
        buf1598 = extern_kernels.convolution(buf1597, buf78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1598, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1599 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_334], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1598, buf1599, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1600 = buf1591; del buf1591  # reuse
        buf1601 = buf1590; del buf1590  # reuse
        buf1602 = buf1589; del buf1589  # reuse
        # Source Nodes: [sp_335], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1599, buf1600, buf1601, buf1602, 1352, 121, grid=grid(1352), stream=stream0)
        buf1603 = buf1593; del buf1593  # reuse
        buf1604 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1606 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_335], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1600, buf1601, buf1602, primals_909, primals_910, buf1603, buf1604, buf1606, primals_909, primals_910, 104, 13, grid=grid(104), stream=stream0)
        del primals_909
        del primals_910
        buf1607 = reinterpret_tensor(buf1609, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2046 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_335, sp_336], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1599, buf1603, buf1604, primals_398, primals_399, buf1607, buf2046, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_399
        buf1608 = reinterpret_tensor(buf1609, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_40], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf1575, buf1608, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1610 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_40], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf1609, buf1610, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf1585
        del buf1596
        del buf1607
        del buf1608
        # Source Nodes: [out_204], Original ATen: [aten.convolution]
        buf1611 = extern_kernels.convolution(buf1610, primals_400, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1611, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1612 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_204], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf1611, buf1612, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1613 = buf1560; del buf1560  # reuse
        buf1614 = buf1559; del buf1559  # reuse
        buf1615 = buf1558; del buf1558  # reuse
        # Source Nodes: [out_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf1612, buf1613, buf1614, buf1615, 13312, 121, grid=grid(13312), stream=stream0)
        buf1616 = buf1562; del buf1562  # reuse
        buf1617 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1619 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf1613, buf1614, buf1615, primals_912, primals_913, buf1616, buf1617, buf1619, primals_912, primals_913, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_912
        del primals_913
        buf1620 = reinterpret_tensor(buf1611, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1611  # reuse
        # Source Nodes: [out_205, out_206, shortcut_29], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf1612, buf1616, buf1617, primals_401, primals_402, buf1565, buf1620, 1605632, grid=grid(1605632), stream=stream0)
        del primals_402
        # Source Nodes: [out_208], Original ATen: [aten.convolution]
        buf1621 = extern_kernels.convolution(buf1620, primals_403, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1621, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf1622 = reinterpret_tensor(buf1609, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1609  # reuse
        # Source Nodes: [out_208], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf1621, buf1622, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf1623 = buf1570; del buf1570  # reuse
        buf1624 = buf1569; del buf1569  # reuse
        buf1625 = buf1568; del buf1568  # reuse
        # Source Nodes: [out_209], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf1622, buf1623, buf1624, buf1625, 5408, 121, grid=grid(5408), stream=stream0)
        buf1626 = buf1572; del buf1572  # reuse
        buf1627 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf1629 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_209], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf1623, buf1624, buf1625, primals_915, primals_916, buf1626, buf1627, buf1629, primals_915, primals_916, 416, 13, grid=grid(416), stream=stream0)
        del primals_915
        del primals_916
        buf1630 = reinterpret_tensor(buf1621, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1621  # reuse
        buf2045 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_209, out_210], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf1622, buf1626, buf1627, primals_404, primals_405, buf1630, buf2045, 652288, grid=grid(652288), stream=stream0)
        del primals_405
        # Source Nodes: [sp_339], Original ATen: [aten.convolution]
        buf1631 = extern_kernels.convolution(reinterpret_tensor(buf1630, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1631, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1632 = reinterpret_tensor(buf1598, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1598  # reuse
        # Source Nodes: [sp_339], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1631, buf1632, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1633 = buf1602; del buf1602  # reuse
        buf1634 = buf1601; del buf1601  # reuse
        buf1635 = buf1600; del buf1600  # reuse
        # Source Nodes: [sp_340], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1632, buf1633, buf1634, buf1635, 1352, 121, grid=grid(1352), stream=stream0)
        buf1636 = buf1604; del buf1604  # reuse
        buf1637 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1639 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_340], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1633, buf1634, buf1635, primals_918, primals_919, buf1636, buf1637, buf1639, primals_918, primals_919, 104, 13, grid=grid(104), stream=stream0)
        del primals_918
        del primals_919
        buf1664 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf1640 = reinterpret_tensor(buf1664, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2044 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_340, sp_341], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1632, buf1636, buf1637, primals_407, primals_408, buf1640, buf2044, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_408
        buf1641 = reinterpret_tensor(buf1631, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1631  # reuse
        # Source Nodes: [sp_342], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf1640, buf1630, buf1641, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_343], Original ATen: [aten.convolution]
        buf1642 = extern_kernels.convolution(buf1641, buf80, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1642, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1643 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_343], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1642, buf1643, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1644 = buf1635; del buf1635  # reuse
        buf1645 = buf1634; del buf1634  # reuse
        buf1646 = buf1633; del buf1633  # reuse
        # Source Nodes: [sp_344], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1643, buf1644, buf1645, buf1646, 1352, 121, grid=grid(1352), stream=stream0)
        buf1647 = buf1637; del buf1637  # reuse
        buf1648 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1650 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_344], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1644, buf1645, buf1646, primals_921, primals_922, buf1647, buf1648, buf1650, primals_921, primals_922, 104, 13, grid=grid(104), stream=stream0)
        del primals_921
        del primals_922
        buf1651 = reinterpret_tensor(buf1664, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2043 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_344, sp_345], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1643, buf1647, buf1648, primals_410, primals_411, buf1651, buf2043, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_411
        buf1652 = reinterpret_tensor(buf1642, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1642  # reuse
        # Source Nodes: [sp_346], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf1651, buf1630, buf1652, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_347], Original ATen: [aten.convolution]
        buf1653 = extern_kernels.convolution(buf1652, buf81, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1653, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1654 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_347], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1653, buf1654, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1655 = buf1646; del buf1646  # reuse
        buf1656 = buf1645; del buf1645  # reuse
        buf1657 = buf1644; del buf1644  # reuse
        # Source Nodes: [sp_348], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1654, buf1655, buf1656, buf1657, 1352, 121, grid=grid(1352), stream=stream0)
        buf1658 = buf1648; del buf1648  # reuse
        buf1659 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1661 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_348], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1655, buf1656, buf1657, primals_924, primals_925, buf1658, buf1659, buf1661, primals_924, primals_925, 104, 13, grid=grid(104), stream=stream0)
        del primals_924
        del primals_925
        buf1662 = reinterpret_tensor(buf1664, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2042 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_348, sp_349], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1654, buf1658, buf1659, primals_413, primals_414, buf1662, buf2042, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_414
        buf1663 = reinterpret_tensor(buf1664, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_39], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf1630, buf1663, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1665 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_39], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf1664, buf1665, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf1640
        del buf1651
        del buf1662
        del buf1663
        # Source Nodes: [out_212], Original ATen: [aten.convolution]
        buf1666 = extern_kernels.convolution(buf1665, primals_415, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1666, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1667 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_212], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf1666, buf1667, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1668 = buf1615; del buf1615  # reuse
        buf1669 = buf1614; del buf1614  # reuse
        buf1670 = buf1613; del buf1613  # reuse
        # Source Nodes: [out_213], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf1667, buf1668, buf1669, buf1670, 13312, 121, grid=grid(13312), stream=stream0)
        buf1671 = buf1617; del buf1617  # reuse
        buf1672 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1674 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_213], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf1668, buf1669, buf1670, primals_927, primals_928, buf1671, buf1672, buf1674, primals_927, primals_928, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_927
        del primals_928
        buf1675 = reinterpret_tensor(buf1666, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1666  # reuse
        # Source Nodes: [out_213, out_214, shortcut_30], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf1667, buf1671, buf1672, primals_416, primals_417, buf1620, buf1675, 1605632, grid=grid(1605632), stream=stream0)
        del primals_417
        # Source Nodes: [out_216], Original ATen: [aten.convolution]
        buf1676 = extern_kernels.convolution(buf1675, primals_418, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1676, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf1677 = reinterpret_tensor(buf1664, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1664  # reuse
        # Source Nodes: [out_216], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf1676, buf1677, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf1678 = buf1625; del buf1625  # reuse
        buf1679 = buf1624; del buf1624  # reuse
        buf1680 = buf1623; del buf1623  # reuse
        # Source Nodes: [out_217], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf1677, buf1678, buf1679, buf1680, 5408, 121, grid=grid(5408), stream=stream0)
        buf1681 = buf1627; del buf1627  # reuse
        buf1682 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf1684 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_217], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf1678, buf1679, buf1680, primals_930, primals_931, buf1681, buf1682, buf1684, primals_930, primals_931, 416, 13, grid=grid(416), stream=stream0)
        del primals_930
        del primals_931
        buf1685 = reinterpret_tensor(buf1676, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1676  # reuse
        buf2041 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_217, out_218], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf1677, buf1681, buf1682, primals_419, primals_420, buf1685, buf2041, 652288, grid=grid(652288), stream=stream0)
        del primals_420
        # Source Nodes: [sp_352], Original ATen: [aten.convolution]
        buf1686 = extern_kernels.convolution(reinterpret_tensor(buf1685, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1686, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1687 = reinterpret_tensor(buf1653, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1653  # reuse
        # Source Nodes: [sp_352], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1686, buf1687, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1688 = buf1657; del buf1657  # reuse
        buf1689 = buf1656; del buf1656  # reuse
        buf1690 = buf1655; del buf1655  # reuse
        # Source Nodes: [sp_353], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1687, buf1688, buf1689, buf1690, 1352, 121, grid=grid(1352), stream=stream0)
        buf1691 = buf1659; del buf1659  # reuse
        buf1692 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1694 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_353], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1688, buf1689, buf1690, primals_933, primals_934, buf1691, buf1692, buf1694, primals_933, primals_934, 104, 13, grid=grid(104), stream=stream0)
        del primals_933
        del primals_934
        buf1719 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf1695 = reinterpret_tensor(buf1719, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2040 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_353, sp_354], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1687, buf1691, buf1692, primals_422, primals_423, buf1695, buf2040, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_423
        buf1696 = reinterpret_tensor(buf1686, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1686  # reuse
        # Source Nodes: [sp_355], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf1695, buf1685, buf1696, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_356], Original ATen: [aten.convolution]
        buf1697 = extern_kernels.convolution(buf1696, buf83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1697, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1698 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_356], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1697, buf1698, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1699 = buf1690; del buf1690  # reuse
        buf1700 = buf1689; del buf1689  # reuse
        buf1701 = buf1688; del buf1688  # reuse
        # Source Nodes: [sp_357], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1698, buf1699, buf1700, buf1701, 1352, 121, grid=grid(1352), stream=stream0)
        buf1702 = buf1692; del buf1692  # reuse
        buf1703 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1705 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_357], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1699, buf1700, buf1701, primals_936, primals_937, buf1702, buf1703, buf1705, primals_936, primals_937, 104, 13, grid=grid(104), stream=stream0)
        del primals_936
        del primals_937
        buf1706 = reinterpret_tensor(buf1719, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2039 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_357, sp_358], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1698, buf1702, buf1703, primals_425, primals_426, buf1706, buf2039, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_426
        buf1707 = reinterpret_tensor(buf1697, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1697  # reuse
        # Source Nodes: [sp_359], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf1706, buf1685, buf1707, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_360], Original ATen: [aten.convolution]
        buf1708 = extern_kernels.convolution(buf1707, buf84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1708, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1709 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_360], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1708, buf1709, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1710 = buf1701; del buf1701  # reuse
        buf1711 = buf1700; del buf1700  # reuse
        buf1712 = buf1699; del buf1699  # reuse
        # Source Nodes: [sp_361], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1709, buf1710, buf1711, buf1712, 1352, 121, grid=grid(1352), stream=stream0)
        buf1713 = buf1703; del buf1703  # reuse
        buf1714 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1716 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_361], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1710, buf1711, buf1712, primals_939, primals_940, buf1713, buf1714, buf1716, primals_939, primals_940, 104, 13, grid=grid(104), stream=stream0)
        del primals_939
        del primals_940
        buf1717 = reinterpret_tensor(buf1719, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2038 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_361, sp_362], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1709, buf1713, buf1714, primals_428, primals_429, buf1717, buf2038, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_429
        buf1718 = reinterpret_tensor(buf1719, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_38], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf1685, buf1718, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1720 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_38], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf1719, buf1720, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf1695
        del buf1706
        del buf1717
        del buf1718
        # Source Nodes: [out_220], Original ATen: [aten.convolution]
        buf1721 = extern_kernels.convolution(buf1720, primals_430, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1721, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1722 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_220], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf1721, buf1722, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1723 = buf1670; del buf1670  # reuse
        buf1724 = buf1669; del buf1669  # reuse
        buf1725 = buf1668; del buf1668  # reuse
        # Source Nodes: [out_221], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf1722, buf1723, buf1724, buf1725, 13312, 121, grid=grid(13312), stream=stream0)
        buf1726 = buf1672; del buf1672  # reuse
        buf1727 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1729 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_221], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf1723, buf1724, buf1725, primals_942, primals_943, buf1726, buf1727, buf1729, primals_942, primals_943, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_942
        del primals_943
        buf1730 = reinterpret_tensor(buf1721, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1721  # reuse
        # Source Nodes: [out_221, out_222, shortcut_31], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf1722, buf1726, buf1727, primals_431, primals_432, buf1675, buf1730, 1605632, grid=grid(1605632), stream=stream0)
        del primals_432
        # Source Nodes: [out_224], Original ATen: [aten.convolution]
        buf1731 = extern_kernels.convolution(buf1730, primals_433, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1731, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf1732 = reinterpret_tensor(buf1719, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1719  # reuse
        # Source Nodes: [out_224], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf1731, buf1732, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf1733 = buf1680; del buf1680  # reuse
        buf1734 = buf1679; del buf1679  # reuse
        buf1735 = buf1678; del buf1678  # reuse
        # Source Nodes: [out_225], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf1732, buf1733, buf1734, buf1735, 5408, 121, grid=grid(5408), stream=stream0)
        buf1736 = buf1682; del buf1682  # reuse
        buf1737 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf1739 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_225], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf1733, buf1734, buf1735, primals_945, primals_946, buf1736, buf1737, buf1739, primals_945, primals_946, 416, 13, grid=grid(416), stream=stream0)
        del primals_945
        del primals_946
        buf1740 = reinterpret_tensor(buf1731, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1731  # reuse
        buf2037 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_225, out_226], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf1732, buf1736, buf1737, primals_434, primals_435, buf1740, buf2037, 652288, grid=grid(652288), stream=stream0)
        del primals_435
        # Source Nodes: [sp_365], Original ATen: [aten.convolution]
        buf1741 = extern_kernels.convolution(reinterpret_tensor(buf1740, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf85, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1741, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1742 = reinterpret_tensor(buf1708, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1708  # reuse
        # Source Nodes: [sp_365], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1741, buf1742, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1743 = buf1712; del buf1712  # reuse
        buf1744 = buf1711; del buf1711  # reuse
        buf1745 = buf1710; del buf1710  # reuse
        # Source Nodes: [sp_366], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1742, buf1743, buf1744, buf1745, 1352, 121, grid=grid(1352), stream=stream0)
        buf1746 = buf1714; del buf1714  # reuse
        buf1747 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1749 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_366], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1743, buf1744, buf1745, primals_948, primals_949, buf1746, buf1747, buf1749, primals_948, primals_949, 104, 13, grid=grid(104), stream=stream0)
        del primals_948
        del primals_949
        buf1774 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf1750 = reinterpret_tensor(buf1774, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2036 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_366, sp_367], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1742, buf1746, buf1747, primals_437, primals_438, buf1750, buf2036, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_438
        buf1751 = reinterpret_tensor(buf1741, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1741  # reuse
        # Source Nodes: [sp_368], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf1750, buf1740, buf1751, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_369], Original ATen: [aten.convolution]
        buf1752 = extern_kernels.convolution(buf1751, buf86, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1752, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1753 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_369], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1752, buf1753, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1754 = buf1745; del buf1745  # reuse
        buf1755 = buf1744; del buf1744  # reuse
        buf1756 = buf1743; del buf1743  # reuse
        # Source Nodes: [sp_370], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1753, buf1754, buf1755, buf1756, 1352, 121, grid=grid(1352), stream=stream0)
        buf1757 = buf1747; del buf1747  # reuse
        buf1758 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1760 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_370], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1754, buf1755, buf1756, primals_951, primals_952, buf1757, buf1758, buf1760, primals_951, primals_952, 104, 13, grid=grid(104), stream=stream0)
        del primals_951
        del primals_952
        buf1761 = reinterpret_tensor(buf1774, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2035 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_370, sp_371], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1753, buf1757, buf1758, primals_440, primals_441, buf1761, buf2035, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_441
        buf1762 = reinterpret_tensor(buf1752, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1752  # reuse
        # Source Nodes: [sp_372], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf1761, buf1740, buf1762, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_373], Original ATen: [aten.convolution]
        buf1763 = extern_kernels.convolution(buf1762, buf87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1763, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1764 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_373], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1763, buf1764, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1765 = buf1756; del buf1756  # reuse
        buf1766 = buf1755; del buf1755  # reuse
        buf1767 = buf1754; del buf1754  # reuse
        # Source Nodes: [sp_374], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1764, buf1765, buf1766, buf1767, 1352, 121, grid=grid(1352), stream=stream0)
        buf1768 = buf1758; del buf1758  # reuse
        buf1769 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1771 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_374], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1765, buf1766, buf1767, primals_954, primals_955, buf1768, buf1769, buf1771, primals_954, primals_955, 104, 13, grid=grid(104), stream=stream0)
        del primals_954
        del primals_955
        buf1772 = reinterpret_tensor(buf1774, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2034 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_374, sp_375], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1764, buf1768, buf1769, primals_443, primals_444, buf1772, buf2034, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_444
        buf1773 = reinterpret_tensor(buf1774, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_37], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf1740, buf1773, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1775 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_37], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf1774, buf1775, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf1750
        del buf1761
        del buf1772
        del buf1773
        # Source Nodes: [out_228], Original ATen: [aten.convolution]
        buf1776 = extern_kernels.convolution(buf1775, primals_445, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1776, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1777 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_228], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf1776, buf1777, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1778 = buf1725; del buf1725  # reuse
        buf1779 = buf1724; del buf1724  # reuse
        buf1780 = buf1723; del buf1723  # reuse
        # Source Nodes: [out_229], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf1777, buf1778, buf1779, buf1780, 13312, 121, grid=grid(13312), stream=stream0)
        buf1781 = buf1727; del buf1727  # reuse
        buf1782 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1784 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_229], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf1778, buf1779, buf1780, primals_957, primals_958, buf1781, buf1782, buf1784, primals_957, primals_958, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_957
        del primals_958
        buf1785 = reinterpret_tensor(buf1776, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1776  # reuse
        # Source Nodes: [out_229, out_230, shortcut_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf1777, buf1781, buf1782, primals_446, primals_447, buf1730, buf1785, 1605632, grid=grid(1605632), stream=stream0)
        del primals_447
        # Source Nodes: [out_232], Original ATen: [aten.convolution]
        buf1786 = extern_kernels.convolution(buf1785, primals_448, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1786, (8, 416, 14, 14), (81536, 196, 14, 1))
        buf1787 = reinterpret_tensor(buf1774, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1774  # reuse
        # Source Nodes: [out_232], Original ATen: [aten.convolution]
        triton_poi_fused_cat_63.run(buf1786, buf1787, 3328, 196, grid=grid(3328, 196), stream=stream0)
        buf1788 = buf1735; del buf1735  # reuse
        buf1789 = buf1734; del buf1734  # reuse
        buf1790 = buf1733; del buf1733  # reuse
        # Source Nodes: [out_233], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf1787, buf1788, buf1789, buf1790, 5408, 121, grid=grid(5408), stream=stream0)
        buf1791 = buf1737; del buf1737  # reuse
        buf1792 = empty_strided((1, 416, 1, 1), (416, 1, 416, 416), device='cuda', dtype=torch.float32)
        buf1794 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_233], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf1788, buf1789, buf1790, primals_960, primals_961, buf1791, buf1792, buf1794, primals_960, primals_961, 416, 13, grid=grid(416), stream=stream0)
        del buf1788
        del buf1789
        del buf1790
        del primals_960
        del primals_961
        buf1795 = reinterpret_tensor(buf1786, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf1786  # reuse
        buf2033 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_233, out_234], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_70.run(buf1787, buf1791, buf1792, primals_449, primals_450, buf1795, buf2033, 652288, grid=grid(652288), stream=stream0)
        del buf1792
        del primals_450
        # Source Nodes: [sp_378], Original ATen: [aten.convolution]
        buf1796 = extern_kernels.convolution(reinterpret_tensor(buf1795, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1796, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1797 = reinterpret_tensor(buf1763, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1763  # reuse
        # Source Nodes: [sp_378], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1796, buf1797, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1798 = buf1767; del buf1767  # reuse
        buf1799 = buf1766; del buf1766  # reuse
        buf1800 = buf1765; del buf1765  # reuse
        # Source Nodes: [sp_379], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1797, buf1798, buf1799, buf1800, 1352, 121, grid=grid(1352), stream=stream0)
        buf1801 = buf1769; del buf1769  # reuse
        buf1802 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1804 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_379], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1798, buf1799, buf1800, primals_963, primals_964, buf1801, buf1802, buf1804, primals_963, primals_964, 104, 13, grid=grid(104), stream=stream0)
        del primals_963
        del primals_964
        buf1829 = empty((8, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf1805 = reinterpret_tensor(buf1829, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf2032 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_379, sp_380], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1797, buf1801, buf1802, primals_452, primals_453, buf1805, buf2032, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_453
        buf1806 = reinterpret_tensor(buf1796, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1796  # reuse
        # Source Nodes: [sp_381], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf1805, buf1795, buf1806, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_382], Original ATen: [aten.convolution]
        buf1807 = extern_kernels.convolution(buf1806, buf89, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1807, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1808 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_382], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1807, buf1808, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1809 = buf1800; del buf1800  # reuse
        buf1810 = buf1799; del buf1799  # reuse
        buf1811 = buf1798; del buf1798  # reuse
        # Source Nodes: [sp_383], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1808, buf1809, buf1810, buf1811, 1352, 121, grid=grid(1352), stream=stream0)
        buf1812 = buf1802; del buf1802  # reuse
        buf1813 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1815 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_383], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1809, buf1810, buf1811, primals_966, primals_967, buf1812, buf1813, buf1815, primals_966, primals_967, 104, 13, grid=grid(104), stream=stream0)
        del primals_966
        del primals_967
        buf1816 = reinterpret_tensor(buf1829, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf2031 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_383, sp_384], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1808, buf1812, buf1813, primals_455, primals_456, buf1816, buf2031, 832, 196, grid=grid(832, 196), stream=stream0)
        del primals_456
        buf1817 = reinterpret_tensor(buf1807, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1807  # reuse
        # Source Nodes: [sp_385], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf1816, buf1795, buf1817, 1568, 104, grid=grid(1568, 104), stream=stream0)
        # Source Nodes: [sp_386], Original ATen: [aten.convolution]
        buf1818 = extern_kernels.convolution(buf1817, buf90, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1818, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf1819 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_386], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf1818, buf1819, 832, 196, grid=grid(832, 196), stream=stream0)
        del buf1818
        buf1820 = buf1811; del buf1811  # reuse
        buf1821 = buf1810; del buf1810  # reuse
        buf1822 = buf1809; del buf1809  # reuse
        # Source Nodes: [sp_387], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf1819, buf1820, buf1821, buf1822, 1352, 121, grid=grid(1352), stream=stream0)
        buf1823 = buf1813; del buf1813  # reuse
        buf1824 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf1826 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_387], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf1820, buf1821, buf1822, primals_969, primals_970, buf1823, buf1824, buf1826, primals_969, primals_970, 104, 13, grid=grid(104), stream=stream0)
        del buf1820
        del buf1821
        del buf1822
        del primals_969
        del primals_970
        buf1827 = reinterpret_tensor(buf1829, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        buf2030 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_387, sp_388], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf1819, buf1823, buf1824, primals_458, primals_459, buf1827, buf2030, 832, 196, grid=grid(832, 196), stream=stream0)
        del buf1824
        del primals_459
        buf1828 = reinterpret_tensor(buf1829, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_36], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf1795, buf1828, 832, 196, grid=grid(832, 196), stream=stream0)
        buf1830 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_36], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf1829, buf1830, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf1805
        del buf1816
        del buf1827
        del buf1828
        del buf1829
        # Source Nodes: [out_236], Original ATen: [aten.convolution]
        buf1831 = extern_kernels.convolution(buf1830, primals_460, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1831, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1832 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_236], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf1831, buf1832, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1833 = buf1780; del buf1780  # reuse
        buf1834 = buf1779; del buf1779  # reuse
        buf1835 = buf1778; del buf1778  # reuse
        # Source Nodes: [out_237], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf1832, buf1833, buf1834, buf1835, 13312, 121, grid=grid(13312), stream=stream0)
        buf1836 = buf1782; del buf1782  # reuse
        buf1837 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1839 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_237], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf1833, buf1834, buf1835, primals_972, primals_973, buf1836, buf1837, buf1839, primals_972, primals_973, 1024, 13, grid=grid(1024), stream=stream0)
        del buf1833
        del buf1834
        del buf1835
        del primals_972
        del primals_973
        buf1840 = reinterpret_tensor(buf1831, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1831  # reuse
        # Source Nodes: [out_237, out_238, shortcut_33], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_74.run(buf1832, buf1836, buf1837, primals_461, primals_462, buf1785, buf1840, 1605632, grid=grid(1605632), stream=stream0)
        del buf1837
        del primals_462
        # Source Nodes: [out_240], Original ATen: [aten.convolution]
        buf1841 = extern_kernels.convolution(buf1840, primals_463, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1841, (8, 832, 14, 14), (163072, 196, 14, 1))
        buf1842 = reinterpret_tensor(buf556, (8, 832, 14, 14), (163072, 1, 11648, 832), 0); del buf556  # reuse
        # Source Nodes: [out_240], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_75.run(buf1841, buf1842, 6656, 196, grid=grid(6656, 196), stream=stream0)
        buf1843 = empty_strided((1, 832, 1, 1, 13), (10816, 1, 10816, 10816, 832), device='cuda', dtype=torch.float32)
        buf1844 = empty_strided((1, 832, 1, 1, 13), (10816, 1, 10816, 10816, 832), device='cuda', dtype=torch.float32)
        buf1845 = empty_strided((1, 832, 1, 1, 13), (10816, 1, 10816, 10816, 832), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_241], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf1842, buf1843, buf1844, buf1845, 10816, 121, grid=grid(10816), stream=stream0)
        buf1846 = empty_strided((1, 832, 1, 1), (832, 1, 832, 832), device='cuda', dtype=torch.float32)
        buf1847 = empty_strided((1, 832, 1, 1), (832, 1, 832, 832), device='cuda', dtype=torch.float32)
        buf1849 = empty((832, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_241], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_77.run(buf1843, buf1844, buf1845, primals_975, primals_976, buf1846, buf1847, buf1849, primals_975, primals_976, 832, 13, grid=grid(832), stream=stream0)
        del buf1843
        del buf1844
        del buf1845
        del primals_975
        del primals_976
        buf1850 = reinterpret_tensor(buf1841, (8, 832, 14, 14), (163072, 1, 11648, 832), 0); del buf1841  # reuse
        buf2029 = empty_strided((8, 832, 14, 14), (163072, 1, 11648, 832), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_241, out_242], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_78.run(buf1842, buf1846, buf1847, primals_464, primals_465, buf1850, buf2029, 1304576, grid=grid(1304576), stream=stream0)
        del primals_465
        # Source Nodes: [sp_391], Original ATen: [aten.convolution]
        buf1851 = extern_kernels.convolution(reinterpret_tensor(buf1850, (8, 208, 14, 14), (163072, 1, 11648, 832), 0), buf91, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1851, (8, 208, 7, 7), (10192, 49, 7, 1))
        buf1852 = empty_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_391], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf1851, buf1852, 1664, 49, grid=grid(1664, 49), stream=stream0)
        buf1853 = reinterpret_tensor(buf1847, (1, 208, 1, 1, 4), (832, 1, 832, 832, 208), 0); del buf1847  # reuse
        buf1854 = empty_strided((1, 208, 1, 1, 4), (832, 1, 832, 832, 208), device='cuda', dtype=torch.float32)
        buf1855 = empty_strided((1, 208, 1, 1, 4), (832, 1, 832, 832, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_392], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_80.run(buf1852, buf1853, buf1854, buf1855, 832, 98, grid=grid(832), stream=stream0)
        buf1856 = buf519; del buf519  # reuse
        buf1857 = empty_strided((1, 208, 1, 1), (208, 1, 208, 208), device='cuda', dtype=torch.float32)
        buf1859 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_392], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_81.run(buf1853, buf1854, buf1855, primals_978, primals_979, buf1856, buf1857, buf1859, primals_978, primals_979, 208, 4, grid=grid(208), stream=stream0)
        del primals_978
        del primals_979
        buf1882 = reinterpret_tensor(buf545, (8, 832, 7, 7), (40768, 49, 7, 1), 0); del buf545  # reuse
        buf1860 = reinterpret_tensor(buf1882, (8, 208, 7, 7), (40768, 49, 7, 1), 0)  # alias
        buf2028 = empty_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_392, sp_393], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_82.run(buf1852, buf1856, buf1857, primals_467, primals_468, buf1860, buf2028, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del primals_468
        # Source Nodes: [sp_395], Original ATen: [aten.convolution]
        buf1861 = extern_kernels.convolution(reinterpret_tensor(buf1850, (8, 208, 14, 14), (163072, 1, 11648, 832), 208), buf92, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1861, (8, 208, 7, 7), (10192, 49, 7, 1))
        buf1862 = reinterpret_tensor(buf1851, (8, 208, 7, 7), (10192, 1, 1456, 208), 0); del buf1851  # reuse
        # Source Nodes: [sp_395], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf1861, buf1862, 1664, 49, grid=grid(1664, 49), stream=stream0)
        buf1863 = buf1855; del buf1855  # reuse
        buf1864 = buf1854; del buf1854  # reuse
        buf1865 = buf1853; del buf1853  # reuse
        # Source Nodes: [sp_396], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_80.run(buf1862, buf1863, buf1864, buf1865, 832, 98, grid=grid(832), stream=stream0)
        buf1866 = buf1857; del buf1857  # reuse
        buf1867 = empty_strided((1, 208, 1, 1), (208, 1, 208, 208), device='cuda', dtype=torch.float32)
        buf1869 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_396], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_81.run(buf1863, buf1864, buf1865, primals_981, primals_982, buf1866, buf1867, buf1869, primals_981, primals_982, 208, 4, grid=grid(208), stream=stream0)
        del primals_981
        del primals_982
        buf1870 = reinterpret_tensor(buf1882, (8, 208, 7, 7), (40768, 49, 7, 1), 10192)  # alias
        buf2027 = empty_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_396, sp_397], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_82.run(buf1862, buf1866, buf1867, primals_470, primals_471, buf1870, buf2027, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del primals_471
        # Source Nodes: [sp_399], Original ATen: [aten.convolution]
        buf1871 = extern_kernels.convolution(reinterpret_tensor(buf1850, (8, 208, 14, 14), (163072, 1, 11648, 832), 416), buf93, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1871, (8, 208, 7, 7), (10192, 49, 7, 1))
        buf1872 = reinterpret_tensor(buf1861, (8, 208, 7, 7), (10192, 1, 1456, 208), 0); del buf1861  # reuse
        # Source Nodes: [sp_399], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf1871, buf1872, 1664, 49, grid=grid(1664, 49), stream=stream0)
        buf1873 = buf1865; del buf1865  # reuse
        buf1874 = buf1864; del buf1864  # reuse
        buf1875 = buf1863; del buf1863  # reuse
        # Source Nodes: [sp_400], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_80.run(buf1872, buf1873, buf1874, buf1875, 832, 98, grid=grid(832), stream=stream0)
        buf1876 = buf1867; del buf1867  # reuse
        buf1877 = empty_strided((1, 208, 1, 1), (208, 1, 208, 208), device='cuda', dtype=torch.float32)
        buf1879 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_400], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_81.run(buf1873, buf1874, buf1875, primals_984, primals_985, buf1876, buf1877, buf1879, primals_984, primals_985, 208, 4, grid=grid(208), stream=stream0)
        del primals_984
        del primals_985
        buf1880 = reinterpret_tensor(buf1882, (8, 208, 7, 7), (40768, 49, 7, 1), 20384)  # alias
        buf2026 = empty_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_400, sp_401], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_82.run(buf1872, buf1876, buf1877, primals_473, primals_474, buf1880, buf2026, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del primals_474
        buf1881 = reinterpret_tensor(buf1882, (8, 208, 7, 7), (40768, 49, 7, 1), 30576)  # alias
        # Source Nodes: [getattr_l__mod___layer4___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_83.run(buf1850, buf1881, 1664, 49, grid=grid(1664, 49), stream=stream0)
        buf1883 = empty_strided((8, 832, 7, 7), (40768, 1, 5824, 832), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_35], Original ATen: [aten.cat]
        triton_poi_fused_cat_84.run(buf1882, buf1883, 6656, 49, grid=grid(6656, 49), stream=stream0)
        del buf1860
        del buf1870
        del buf1880
        del buf1881
        # Source Nodes: [out_244], Original ATen: [aten.convolution]
        buf1884 = extern_kernels.convolution(buf1883, primals_475, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1884, (8, 2048, 7, 7), (100352, 49, 7, 1))
        buf1885 = empty_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_244], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_85.run(buf1884, buf1885, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf1886 = empty_strided((1, 2048, 1, 1, 4), (8192, 1, 8192, 8192, 2048), device='cuda', dtype=torch.float32)
        buf1887 = empty_strided((1, 2048, 1, 1, 4), (8192, 1, 8192, 8192, 2048), device='cuda', dtype=torch.float32)
        buf1888 = empty_strided((1, 2048, 1, 1, 4), (8192, 1, 8192, 8192, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_245], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf1885, buf1886, buf1887, buf1888, 8192, 98, grid=grid(8192), stream=stream0)
        buf1889 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf1890 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf1892 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_245], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf1886, buf1887, buf1888, primals_987, primals_988, buf1889, buf1890, buf1892, primals_987, primals_988, 2048, 4, grid=grid(2048), stream=stream0)
        del primals_987
        del primals_988
        # Source Nodes: [getattr_l__mod___layer4___0___downsample_0], Original ATen: [aten.convolution]
        buf1893 = extern_kernels.convolution(buf1840, primals_478, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1893, (8, 2048, 7, 7), (100352, 49, 7, 1))
        buf1894 = reinterpret_tensor(buf1884, (8, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf1884  # reuse
        # Source Nodes: [getattr_l__mod___layer4___0___downsample_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_85.run(buf1893, buf1894, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf1895 = buf1888; del buf1888  # reuse
        buf1896 = buf1887; del buf1887  # reuse
        buf1897 = buf1886; del buf1886  # reuse
        # Source Nodes: [shortcut_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf1894, buf1895, buf1896, buf1897, 8192, 98, grid=grid(8192), stream=stream0)
        buf1898 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf1899 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf1901 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf1895, buf1896, buf1897, primals_990, primals_991, buf1898, buf1899, buf1901, primals_990, primals_991, 2048, 4, grid=grid(2048), stream=stream0)
        del primals_990
        del primals_991
        buf1902 = reinterpret_tensor(buf1893, (8, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf1893  # reuse
        buf1903 = buf1902; del buf1902  # reuse
        # Source Nodes: [out_245, out_246, shortcut_34, shortcut_35], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_88.run(buf1903, buf1885, buf1889, buf1890, primals_476, primals_477, buf1894, buf1898, buf1899, primals_479, primals_480, 802816, grid=grid(802816), stream=stream0)
        del primals_477
        del primals_480
        # Source Nodes: [out_248], Original ATen: [aten.convolution]
        buf1904 = extern_kernels.convolution(buf1903, primals_481, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1904, (8, 832, 7, 7), (40768, 49, 7, 1))
        buf1905 = reinterpret_tensor(buf1882, (8, 832, 7, 7), (40768, 1, 5824, 832), 0); del buf1882  # reuse
        # Source Nodes: [out_248], Original ATen: [aten.convolution]
        triton_poi_fused_cat_84.run(buf1904, buf1905, 6656, 49, grid=grid(6656, 49), stream=stream0)
        buf1906 = empty_strided((1, 832, 1, 1, 4), (3328, 1, 3328, 3328, 832), device='cuda', dtype=torch.float32)
        buf1907 = empty_strided((1, 832, 1, 1, 4), (3328, 1, 3328, 3328, 832), device='cuda', dtype=torch.float32)
        buf1908 = empty_strided((1, 832, 1, 1, 4), (3328, 1, 3328, 3328, 832), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_249], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_89.run(buf1905, buf1906, buf1907, buf1908, 3328, 98, grid=grid(3328), stream=stream0)
        buf1909 = reinterpret_tensor(buf1875, (1, 832, 1, 1), (832, 1, 832, 832), 0); del buf1875  # reuse
        buf1910 = reinterpret_tensor(buf1874, (1, 832, 1, 1), (832, 1, 832, 832), 0); del buf1874  # reuse
        buf1912 = reinterpret_tensor(buf1873, (832, ), (1, ), 0); del buf1873  # reuse
        # Source Nodes: [out_249], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_90.run(buf1906, buf1907, buf1908, primals_993, primals_994, buf1909, buf1910, buf1912, primals_993, primals_994, 832, 4, grid=grid(832), stream=stream0)
        del primals_993
        del primals_994
        buf1913 = reinterpret_tensor(buf1904, (8, 832, 7, 7), (40768, 1, 5824, 832), 0); del buf1904  # reuse
        buf2025 = empty_strided((8, 832, 7, 7), (40768, 1, 5824, 832), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_249, out_250], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_91.run(buf1905, buf1909, buf1910, primals_482, primals_483, buf1913, buf2025, 326144, grid=grid(326144), stream=stream0)
        del primals_483
        # Source Nodes: [sp_404], Original ATen: [aten.convolution]
        buf1914 = extern_kernels.convolution(reinterpret_tensor(buf1913, (8, 208, 7, 7), (40768, 1, 5824, 832), 0), buf94, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1914, (8, 208, 7, 7), (10192, 49, 7, 1))
        buf1915 = reinterpret_tensor(buf1871, (8, 208, 7, 7), (10192, 1, 1456, 208), 0); del buf1871  # reuse
        # Source Nodes: [sp_404], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf1914, buf1915, 1664, 49, grid=grid(1664, 49), stream=stream0)
        buf1916 = reinterpret_tensor(buf1910, (1, 208, 1, 1, 4), (832, 1, 832, 832, 208), 0); del buf1910  # reuse
        buf1917 = empty_strided((1, 208, 1, 1, 4), (832, 1, 832, 832, 208), device='cuda', dtype=torch.float32)
        buf1918 = empty_strided((1, 208, 1, 1, 4), (832, 1, 832, 832, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_405], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_80.run(buf1915, buf1916, buf1917, buf1918, 832, 98, grid=grid(832), stream=stream0)
        buf1919 = buf1877; del buf1877  # reuse
        buf1920 = empty_strided((1, 208, 1, 1), (208, 1, 208, 208), device='cuda', dtype=torch.float32)
        buf1922 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_405], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_81.run(buf1916, buf1917, buf1918, primals_996, primals_997, buf1919, buf1920, buf1922, primals_996, primals_997, 208, 4, grid=grid(208), stream=stream0)
        del primals_996
        del primals_997
        buf1947 = empty((8, 832, 7, 7), device='cuda', dtype=torch.float32)
        buf1923 = reinterpret_tensor(buf1947, (8, 208, 7, 7), (40768, 49, 7, 1), 0)  # alias
        buf2024 = empty_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_405, sp_406], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_82.run(buf1915, buf1919, buf1920, primals_485, primals_486, buf1923, buf2024, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del primals_486
        buf1924 = reinterpret_tensor(buf1914, (8, 208, 7, 7), (10192, 1, 1456, 208), 0); del buf1914  # reuse
        # Source Nodes: [sp_407], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(buf1923, buf1913, buf1924, 392, 208, grid=grid(392, 208), stream=stream0)
        # Source Nodes: [sp_408], Original ATen: [aten.convolution]
        buf1925 = extern_kernels.convolution(buf1924, buf95, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1925, (8, 208, 7, 7), (10192, 49, 7, 1))
        buf1926 = empty_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_408], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf1925, buf1926, 1664, 49, grid=grid(1664, 49), stream=stream0)
        buf1927 = buf1918; del buf1918  # reuse
        buf1928 = buf1917; del buf1917  # reuse
        buf1929 = buf1916; del buf1916  # reuse
        # Source Nodes: [sp_409], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_80.run(buf1926, buf1927, buf1928, buf1929, 832, 98, grid=grid(832), stream=stream0)
        buf1930 = buf1920; del buf1920  # reuse
        buf1931 = empty_strided((1, 208, 1, 1), (208, 1, 208, 208), device='cuda', dtype=torch.float32)
        buf1933 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_409], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_81.run(buf1927, buf1928, buf1929, primals_999, primals_1000, buf1930, buf1931, buf1933, primals_999, primals_1000, 208, 4, grid=grid(208), stream=stream0)
        del primals_1000
        del primals_999
        buf1934 = reinterpret_tensor(buf1947, (8, 208, 7, 7), (40768, 49, 7, 1), 10192)  # alias
        buf2023 = empty_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_409, sp_410], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_82.run(buf1926, buf1930, buf1931, primals_488, primals_489, buf1934, buf2023, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del primals_489
        buf1935 = reinterpret_tensor(buf1925, (8, 208, 7, 7), (10192, 1, 1456, 208), 0); del buf1925  # reuse
        # Source Nodes: [sp_411], Original ATen: [aten.add]
        triton_poi_fused_add_93.run(buf1934, buf1913, buf1935, 392, 208, grid=grid(392, 208), stream=stream0)
        # Source Nodes: [sp_412], Original ATen: [aten.convolution]
        buf1936 = extern_kernels.convolution(buf1935, buf96, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1936, (8, 208, 7, 7), (10192, 49, 7, 1))
        buf1937 = empty_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_412], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf1936, buf1937, 1664, 49, grid=grid(1664, 49), stream=stream0)
        buf1938 = buf1929; del buf1929  # reuse
        buf1939 = buf1928; del buf1928  # reuse
        buf1940 = buf1927; del buf1927  # reuse
        # Source Nodes: [sp_413], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_80.run(buf1937, buf1938, buf1939, buf1940, 832, 98, grid=grid(832), stream=stream0)
        buf1941 = buf1931; del buf1931  # reuse
        buf1942 = empty_strided((1, 208, 1, 1), (208, 1, 208, 208), device='cuda', dtype=torch.float32)
        buf1944 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_413], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_81.run(buf1938, buf1939, buf1940, primals_1002, primals_1003, buf1941, buf1942, buf1944, primals_1002, primals_1003, 208, 4, grid=grid(208), stream=stream0)
        del primals_1002
        del primals_1003
        buf1945 = reinterpret_tensor(buf1947, (8, 208, 7, 7), (40768, 49, 7, 1), 20384)  # alias
        buf2022 = empty_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_413, sp_414], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_82.run(buf1937, buf1941, buf1942, primals_491, primals_492, buf1945, buf2022, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del primals_492
        buf1946 = reinterpret_tensor(buf1947, (8, 208, 7, 7), (40768, 49, 7, 1), 30576)  # alias
        # Source Nodes: [cat_34], Original ATen: [aten.cat]
        triton_poi_fused_cat_94.run(buf1913, buf1946, 1664, 49, grid=grid(1664, 49), stream=stream0)
        buf1948 = empty_strided((8, 832, 7, 7), (40768, 1, 5824, 832), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_34], Original ATen: [aten.cat]
        triton_poi_fused_cat_84.run(buf1947, buf1948, 6656, 49, grid=grid(6656, 49), stream=stream0)
        del buf1923
        del buf1934
        del buf1945
        del buf1946
        # Source Nodes: [out_252], Original ATen: [aten.convolution]
        buf1949 = extern_kernels.convolution(buf1948, primals_493, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1949, (8, 2048, 7, 7), (100352, 49, 7, 1))
        buf1950 = empty_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_252], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_85.run(buf1949, buf1950, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf1951 = buf1897; del buf1897  # reuse
        buf1952 = buf1896; del buf1896  # reuse
        buf1953 = buf1895; del buf1895  # reuse
        # Source Nodes: [out_253], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf1950, buf1951, buf1952, buf1953, 8192, 98, grid=grid(8192), stream=stream0)
        buf1954 = buf1899; del buf1899  # reuse
        buf1955 = buf1890; del buf1890  # reuse
        buf1957 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_253], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf1951, buf1952, buf1953, primals_1005, primals_1006, buf1954, buf1955, buf1957, primals_1005, primals_1006, 2048, 4, grid=grid(2048), stream=stream0)
        del primals_1005
        del primals_1006
        buf1958 = reinterpret_tensor(buf1949, (8, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf1949  # reuse
        # Source Nodes: [out_253, out_254, shortcut_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_95.run(buf1950, buf1954, buf1955, primals_494, primals_495, buf1903, buf1958, 802816, grid=grid(802816), stream=stream0)
        del primals_495
        # Source Nodes: [out_256], Original ATen: [aten.convolution]
        buf1959 = extern_kernels.convolution(buf1958, primals_496, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1959, (8, 832, 7, 7), (40768, 49, 7, 1))
        buf1960 = reinterpret_tensor(buf1947, (8, 832, 7, 7), (40768, 1, 5824, 832), 0); del buf1947  # reuse
        # Source Nodes: [out_256], Original ATen: [aten.convolution]
        triton_poi_fused_cat_84.run(buf1959, buf1960, 6656, 49, grid=grid(6656, 49), stream=stream0)
        buf1961 = buf1908; del buf1908  # reuse
        buf1962 = buf1907; del buf1907  # reuse
        buf1963 = buf1906; del buf1906  # reuse
        # Source Nodes: [out_257], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_89.run(buf1960, buf1961, buf1962, buf1963, 3328, 98, grid=grid(3328), stream=stream0)
        buf1964 = reinterpret_tensor(buf1940, (1, 832, 1, 1), (832, 1, 832, 832), 0); del buf1940  # reuse
        buf1965 = reinterpret_tensor(buf1939, (1, 832, 1, 1), (832, 1, 832, 832), 0); del buf1939  # reuse
        buf1967 = reinterpret_tensor(buf1938, (832, ), (1, ), 0); del buf1938  # reuse
        # Source Nodes: [out_257], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_90.run(buf1961, buf1962, buf1963, primals_1008, primals_1009, buf1964, buf1965, buf1967, primals_1008, primals_1009, 832, 4, grid=grid(832), stream=stream0)
        del buf1961
        del buf1962
        del buf1963
        del primals_1008
        del primals_1009
        buf1968 = reinterpret_tensor(buf1959, (8, 832, 7, 7), (40768, 1, 5824, 832), 0); del buf1959  # reuse
        buf2021 = empty_strided((8, 832, 7, 7), (40768, 1, 5824, 832), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_257, out_258], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_91.run(buf1960, buf1964, buf1965, primals_497, primals_498, buf1968, buf2021, 326144, grid=grid(326144), stream=stream0)
        del primals_498
        # Source Nodes: [sp_417], Original ATen: [aten.convolution]
        buf1969 = extern_kernels.convolution(reinterpret_tensor(buf1968, (8, 208, 7, 7), (40768, 1, 5824, 832), 0), buf97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1969, (8, 208, 7, 7), (10192, 49, 7, 1))
        buf1970 = reinterpret_tensor(buf1936, (8, 208, 7, 7), (10192, 1, 1456, 208), 0); del buf1936  # reuse
        # Source Nodes: [sp_417], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf1969, buf1970, 1664, 49, grid=grid(1664, 49), stream=stream0)
        buf1971 = reinterpret_tensor(buf1965, (1, 208, 1, 1, 4), (832, 1, 832, 832, 208), 0); del buf1965  # reuse
        buf1972 = empty_strided((1, 208, 1, 1, 4), (832, 1, 832, 832, 208), device='cuda', dtype=torch.float32)
        buf1973 = empty_strided((1, 208, 1, 1, 4), (832, 1, 832, 832, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_418], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_80.run(buf1970, buf1971, buf1972, buf1973, 832, 98, grid=grid(832), stream=stream0)
        buf1974 = buf1942; del buf1942  # reuse
        buf1975 = empty_strided((1, 208, 1, 1), (208, 1, 208, 208), device='cuda', dtype=torch.float32)
        buf1977 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_418], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_81.run(buf1971, buf1972, buf1973, primals_1011, primals_1012, buf1974, buf1975, buf1977, primals_1011, primals_1012, 208, 4, grid=grid(208), stream=stream0)
        del primals_1011
        del primals_1012
        buf2002 = empty((8, 832, 7, 7), device='cuda', dtype=torch.float32)
        buf1978 = reinterpret_tensor(buf2002, (8, 208, 7, 7), (40768, 49, 7, 1), 0)  # alias
        buf2020 = empty_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_418, sp_419], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_82.run(buf1970, buf1974, buf1975, primals_500, primals_501, buf1978, buf2020, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del primals_501
        buf1979 = reinterpret_tensor(buf1969, (8, 208, 7, 7), (10192, 1, 1456, 208), 0); del buf1969  # reuse
        # Source Nodes: [sp_420], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(buf1978, buf1968, buf1979, 392, 208, grid=grid(392, 208), stream=stream0)
        # Source Nodes: [sp_421], Original ATen: [aten.convolution]
        buf1980 = extern_kernels.convolution(buf1979, buf98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1980, (8, 208, 7, 7), (10192, 49, 7, 1))
        buf1981 = empty_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_421], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf1980, buf1981, 1664, 49, grid=grid(1664, 49), stream=stream0)
        buf1982 = buf1973; del buf1973  # reuse
        buf1983 = buf1972; del buf1972  # reuse
        buf1984 = buf1971; del buf1971  # reuse
        # Source Nodes: [sp_422], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_80.run(buf1981, buf1982, buf1983, buf1984, 832, 98, grid=grid(832), stream=stream0)
        buf1985 = buf1975; del buf1975  # reuse
        buf1986 = empty_strided((1, 208, 1, 1), (208, 1, 208, 208), device='cuda', dtype=torch.float32)
        buf1988 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_422], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_81.run(buf1982, buf1983, buf1984, primals_1014, primals_1015, buf1985, buf1986, buf1988, primals_1014, primals_1015, 208, 4, grid=grid(208), stream=stream0)
        del primals_1014
        del primals_1015
        buf1989 = reinterpret_tensor(buf2002, (8, 208, 7, 7), (40768, 49, 7, 1), 10192)  # alias
        buf2019 = empty_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_422, sp_423], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_82.run(buf1981, buf1985, buf1986, primals_503, primals_504, buf1989, buf2019, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del primals_504
        buf1990 = reinterpret_tensor(buf1980, (8, 208, 7, 7), (10192, 1, 1456, 208), 0); del buf1980  # reuse
        # Source Nodes: [sp_424], Original ATen: [aten.add]
        triton_poi_fused_add_93.run(buf1989, buf1968, buf1990, 392, 208, grid=grid(392, 208), stream=stream0)
        # Source Nodes: [sp_425], Original ATen: [aten.convolution]
        buf1991 = extern_kernels.convolution(buf1990, buf99, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1991, (8, 208, 7, 7), (10192, 49, 7, 1))
        buf1992 = empty_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_425], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf1991, buf1992, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del buf1991
        buf1993 = buf1984; del buf1984  # reuse
        buf1994 = buf1983; del buf1983  # reuse
        buf1995 = buf1982; del buf1982  # reuse
        # Source Nodes: [sp_426], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_80.run(buf1992, buf1993, buf1994, buf1995, 832, 98, grid=grid(832), stream=stream0)
        buf1996 = buf1986; del buf1986  # reuse
        buf1997 = empty_strided((1, 208, 1, 1), (208, 1, 208, 208), device='cuda', dtype=torch.float32)
        buf1999 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_426], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_81.run(buf1993, buf1994, buf1995, primals_1017, primals_1018, buf1996, buf1997, buf1999, primals_1017, primals_1018, 208, 4, grid=grid(208), stream=stream0)
        del buf1993
        del buf1994
        del buf1995
        del primals_1017
        del primals_1018
        buf2000 = reinterpret_tensor(buf2002, (8, 208, 7, 7), (40768, 49, 7, 1), 20384)  # alias
        buf2018 = empty_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_426, sp_427], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_82.run(buf1992, buf1996, buf1997, primals_506, primals_507, buf2000, buf2018, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del buf1997
        del primals_507
        buf2001 = reinterpret_tensor(buf2002, (8, 208, 7, 7), (40768, 49, 7, 1), 30576)  # alias
        # Source Nodes: [cat_33], Original ATen: [aten.cat]
        triton_poi_fused_cat_94.run(buf1968, buf2001, 1664, 49, grid=grid(1664, 49), stream=stream0)
        buf2003 = empty_strided((8, 832, 7, 7), (40768, 1, 5824, 832), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_33], Original ATen: [aten.cat]
        triton_poi_fused_cat_84.run(buf2002, buf2003, 6656, 49, grid=grid(6656, 49), stream=stream0)
        del buf1978
        del buf1989
        del buf2000
        del buf2001
        del buf2002
        # Source Nodes: [out_260], Original ATen: [aten.convolution]
        buf2004 = extern_kernels.convolution(buf2003, primals_508, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2004, (8, 2048, 7, 7), (100352, 49, 7, 1))
        buf2005 = empty_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_260], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_85.run(buf2004, buf2005, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf2006 = buf1953; del buf1953  # reuse
        buf2007 = buf1952; del buf1952  # reuse
        buf2008 = buf1951; del buf1951  # reuse
        # Source Nodes: [out_261], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf2005, buf2006, buf2007, buf2008, 8192, 98, grid=grid(8192), stream=stream0)
        buf2009 = buf1955; del buf1955  # reuse
        buf2010 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf2012 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_261], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf2006, buf2007, buf2008, primals_1020, primals_1021, buf2009, buf2010, buf2012, primals_1020, primals_1021, 2048, 4, grid=grid(2048), stream=stream0)
        del buf2006
        del buf2007
        del buf2008
        del primals_1020
        del primals_1021
        buf2013 = reinterpret_tensor(buf2004, (8, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf2004  # reuse
        buf2017 = empty_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_261, out_262, x_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_96.run(buf2005, buf2009, buf2010, primals_509, primals_510, buf1958, buf2013, buf2017, 802816, grid=grid(802816), stream=stream0)
        del buf2010
        del primals_510
        buf2014 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cuda', dtype=torch.float32)
        buf2015 = reinterpret_tensor(buf2014, (8, 2048), (2048, 1), 0); del buf2014  # reuse
        # Source Nodes: [x_11, x_9], Original ATen: [aten.mean, aten.view]
        triton_per_fused_mean_view_97.run(buf2015, buf2013, 16384, 49, grid=grid(16384), stream=stream0)
        del buf2013
        buf2016 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_512, buf2015, reinterpret_tensor(primals_511, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf2016)
        del primals_512
        # Source Nodes: [x_1], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_515, primals_515, 1, grid=grid(1), stream=stream0)
        del primals_515
        # Source Nodes: [out_1], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_518, primals_518, 1, grid=grid(1), stream=stream0)
        del primals_518
        # Source Nodes: [sp_2], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_521, primals_521, 1, grid=grid(1), stream=stream0)
        del primals_521
        # Source Nodes: [sp_6], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_524, primals_524, 1, grid=grid(1), stream=stream0)
        del primals_524
        # Source Nodes: [sp_10], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_527, primals_527, 1, grid=grid(1), stream=stream0)
        del primals_527
        # Source Nodes: [out_5], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_530, primals_530, 1, grid=grid(1), stream=stream0)
        del primals_530
        # Source Nodes: [shortcut_1], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_533, primals_533, 1, grid=grid(1), stream=stream0)
        del primals_533
        # Source Nodes: [out_9], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_536, primals_536, 1, grid=grid(1), stream=stream0)
        del primals_536
        # Source Nodes: [sp_15], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_539, primals_539, 1, grid=grid(1), stream=stream0)
        del primals_539
        # Source Nodes: [sp_19], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_542, primals_542, 1, grid=grid(1), stream=stream0)
        del primals_542
        # Source Nodes: [sp_23], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_545, primals_545, 1, grid=grid(1), stream=stream0)
        del primals_545
        # Source Nodes: [out_13], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_548, primals_548, 1, grid=grid(1), stream=stream0)
        del primals_548
        # Source Nodes: [out_17], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_551, primals_551, 1, grid=grid(1), stream=stream0)
        del primals_551
        # Source Nodes: [sp_28], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_554, primals_554, 1, grid=grid(1), stream=stream0)
        del primals_554
        # Source Nodes: [sp_32], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_557, primals_557, 1, grid=grid(1), stream=stream0)
        del primals_557
        # Source Nodes: [sp_36], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_560, primals_560, 1, grid=grid(1), stream=stream0)
        del primals_560
        # Source Nodes: [out_21], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_563, primals_563, 1, grid=grid(1), stream=stream0)
        del primals_563
        # Source Nodes: [out_25], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_566, primals_566, 1, grid=grid(1), stream=stream0)
        del primals_566
        # Source Nodes: [sp_41], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_569, primals_569, 1, grid=grid(1), stream=stream0)
        del primals_569
        # Source Nodes: [sp_45], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_572, primals_572, 1, grid=grid(1), stream=stream0)
        del primals_572
        # Source Nodes: [sp_49], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_575, primals_575, 1, grid=grid(1), stream=stream0)
        del primals_575
        # Source Nodes: [out_29], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_578, primals_578, 1, grid=grid(1), stream=stream0)
        del primals_578
        # Source Nodes: [shortcut_5], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_581, primals_581, 1, grid=grid(1), stream=stream0)
        del primals_581
        # Source Nodes: [out_33], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_584, primals_584, 1, grid=grid(1), stream=stream0)
        del primals_584
        # Source Nodes: [sp_54], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_587, primals_587, 1, grid=grid(1), stream=stream0)
        del primals_587
        # Source Nodes: [sp_58], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_590, primals_590, 1, grid=grid(1), stream=stream0)
        del primals_590
        # Source Nodes: [sp_62], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_593, primals_593, 1, grid=grid(1), stream=stream0)
        del primals_593
        # Source Nodes: [out_37], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_596, primals_596, 1, grid=grid(1), stream=stream0)
        del primals_596
        # Source Nodes: [out_41], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_599, primals_599, 1, grid=grid(1), stream=stream0)
        del primals_599
        # Source Nodes: [sp_67], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_602, primals_602, 1, grid=grid(1), stream=stream0)
        del primals_602
        # Source Nodes: [sp_71], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_605, primals_605, 1, grid=grid(1), stream=stream0)
        del primals_605
        # Source Nodes: [sp_75], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_608, primals_608, 1, grid=grid(1), stream=stream0)
        del primals_608
        # Source Nodes: [out_45], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_611, primals_611, 1, grid=grid(1), stream=stream0)
        del primals_611
        # Source Nodes: [out_49], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_614, primals_614, 1, grid=grid(1), stream=stream0)
        del primals_614
        # Source Nodes: [sp_80], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_617, primals_617, 1, grid=grid(1), stream=stream0)
        del primals_617
        # Source Nodes: [sp_84], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_620, primals_620, 1, grid=grid(1), stream=stream0)
        del primals_620
        # Source Nodes: [sp_88], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_623, primals_623, 1, grid=grid(1), stream=stream0)
        del primals_623
        # Source Nodes: [out_53], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_626, primals_626, 1, grid=grid(1), stream=stream0)
        del primals_626
        # Source Nodes: [out_57], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_629, primals_629, 1, grid=grid(1), stream=stream0)
        del primals_629
        # Source Nodes: [sp_93], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_632, primals_632, 1, grid=grid(1), stream=stream0)
        del primals_632
        # Source Nodes: [sp_97], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_635, primals_635, 1, grid=grid(1), stream=stream0)
        del primals_635
        # Source Nodes: [sp_101], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_638, primals_638, 1, grid=grid(1), stream=stream0)
        del primals_638
        # Source Nodes: [out_61], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_641, primals_641, 1, grid=grid(1), stream=stream0)
        del primals_641
        # Source Nodes: [shortcut_10], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_644, primals_644, 1, grid=grid(1), stream=stream0)
        del primals_644
        # Source Nodes: [out_65], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_647, primals_647, 1, grid=grid(1), stream=stream0)
        del primals_647
        # Source Nodes: [sp_106], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_650, primals_650, 1, grid=grid(1), stream=stream0)
        del primals_650
        # Source Nodes: [sp_110], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_653, primals_653, 1, grid=grid(1), stream=stream0)
        del primals_653
        # Source Nodes: [sp_114], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_656, primals_656, 1, grid=grid(1), stream=stream0)
        del primals_656
        # Source Nodes: [out_69], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_659, primals_659, 1, grid=grid(1), stream=stream0)
        del primals_659
        # Source Nodes: [out_73], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_662, primals_662, 1, grid=grid(1), stream=stream0)
        del primals_662
        # Source Nodes: [sp_119], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_665, primals_665, 1, grid=grid(1), stream=stream0)
        del primals_665
        # Source Nodes: [sp_123], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_668, primals_668, 1, grid=grid(1), stream=stream0)
        del primals_668
        # Source Nodes: [sp_127], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_671, primals_671, 1, grid=grid(1), stream=stream0)
        del primals_671
        # Source Nodes: [out_77], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_674, primals_674, 1, grid=grid(1), stream=stream0)
        del primals_674
        # Source Nodes: [out_81], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_677, primals_677, 1, grid=grid(1), stream=stream0)
        del primals_677
        # Source Nodes: [sp_132], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_680, primals_680, 1, grid=grid(1), stream=stream0)
        del primals_680
        # Source Nodes: [sp_136], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_683, primals_683, 1, grid=grid(1), stream=stream0)
        del primals_683
        # Source Nodes: [sp_140], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_686, primals_686, 1, grid=grid(1), stream=stream0)
        del primals_686
        # Source Nodes: [out_85], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_689, primals_689, 1, grid=grid(1), stream=stream0)
        del primals_689
        # Source Nodes: [out_89], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_692, primals_692, 1, grid=grid(1), stream=stream0)
        del primals_692
        # Source Nodes: [sp_145], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_695, primals_695, 1, grid=grid(1), stream=stream0)
        del primals_695
        # Source Nodes: [sp_149], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_698, primals_698, 1, grid=grid(1), stream=stream0)
        del primals_698
        # Source Nodes: [sp_153], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_701, primals_701, 1, grid=grid(1), stream=stream0)
        del primals_701
        # Source Nodes: [out_93], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_704, primals_704, 1, grid=grid(1), stream=stream0)
        del primals_704
        # Source Nodes: [out_97], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_707, primals_707, 1, grid=grid(1), stream=stream0)
        del primals_707
        # Source Nodes: [sp_158], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_710, primals_710, 1, grid=grid(1), stream=stream0)
        del primals_710
        # Source Nodes: [sp_162], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_713, primals_713, 1, grid=grid(1), stream=stream0)
        del primals_713
        # Source Nodes: [sp_166], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_716, primals_716, 1, grid=grid(1), stream=stream0)
        del primals_716
        # Source Nodes: [out_101], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_719, primals_719, 1, grid=grid(1), stream=stream0)
        del primals_719
        # Source Nodes: [out_105], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_722, primals_722, 1, grid=grid(1), stream=stream0)
        del primals_722
        # Source Nodes: [sp_171], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_725, primals_725, 1, grid=grid(1), stream=stream0)
        del primals_725
        # Source Nodes: [sp_175], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_728, primals_728, 1, grid=grid(1), stream=stream0)
        del primals_728
        # Source Nodes: [sp_179], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_731, primals_731, 1, grid=grid(1), stream=stream0)
        del primals_731
        # Source Nodes: [out_109], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_734, primals_734, 1, grid=grid(1), stream=stream0)
        del primals_734
        # Source Nodes: [out_113], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_737, primals_737, 1, grid=grid(1), stream=stream0)
        del primals_737
        # Source Nodes: [sp_184], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_740, primals_740, 1, grid=grid(1), stream=stream0)
        del primals_740
        # Source Nodes: [sp_188], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_743, primals_743, 1, grid=grid(1), stream=stream0)
        del primals_743
        # Source Nodes: [sp_192], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_746, primals_746, 1, grid=grid(1), stream=stream0)
        del primals_746
        # Source Nodes: [out_117], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_749, primals_749, 1, grid=grid(1), stream=stream0)
        del primals_749
        # Source Nodes: [out_121], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_752, primals_752, 1, grid=grid(1), stream=stream0)
        del primals_752
        # Source Nodes: [sp_197], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_755, primals_755, 1, grid=grid(1), stream=stream0)
        del primals_755
        # Source Nodes: [sp_201], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_758, primals_758, 1, grid=grid(1), stream=stream0)
        del primals_758
        # Source Nodes: [sp_205], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_761, primals_761, 1, grid=grid(1), stream=stream0)
        del primals_761
        # Source Nodes: [out_125], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_764, primals_764, 1, grid=grid(1), stream=stream0)
        del primals_764
        # Source Nodes: [out_129], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_767, primals_767, 1, grid=grid(1), stream=stream0)
        del primals_767
        # Source Nodes: [sp_210], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_770, primals_770, 1, grid=grid(1), stream=stream0)
        del primals_770
        # Source Nodes: [sp_214], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_773, primals_773, 1, grid=grid(1), stream=stream0)
        del primals_773
        # Source Nodes: [sp_218], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_776, primals_776, 1, grid=grid(1), stream=stream0)
        del primals_776
        # Source Nodes: [out_133], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_779, primals_779, 1, grid=grid(1), stream=stream0)
        del primals_779
        # Source Nodes: [out_137], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_782, primals_782, 1, grid=grid(1), stream=stream0)
        del primals_782
        # Source Nodes: [sp_223], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_785, primals_785, 1, grid=grid(1), stream=stream0)
        del primals_785
        # Source Nodes: [sp_227], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_788, primals_788, 1, grid=grid(1), stream=stream0)
        del primals_788
        # Source Nodes: [sp_231], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_791, primals_791, 1, grid=grid(1), stream=stream0)
        del primals_791
        # Source Nodes: [out_141], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_794, primals_794, 1, grid=grid(1), stream=stream0)
        del primals_794
        # Source Nodes: [out_145], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_797, primals_797, 1, grid=grid(1), stream=stream0)
        del primals_797
        # Source Nodes: [sp_236], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_800, primals_800, 1, grid=grid(1), stream=stream0)
        del primals_800
        # Source Nodes: [sp_240], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_803, primals_803, 1, grid=grid(1), stream=stream0)
        del primals_803
        # Source Nodes: [sp_244], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_806, primals_806, 1, grid=grid(1), stream=stream0)
        del primals_806
        # Source Nodes: [out_149], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_809, primals_809, 1, grid=grid(1), stream=stream0)
        del primals_809
        # Source Nodes: [out_153], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_812, primals_812, 1, grid=grid(1), stream=stream0)
        del primals_812
        # Source Nodes: [sp_249], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_815, primals_815, 1, grid=grid(1), stream=stream0)
        del primals_815
        # Source Nodes: [sp_253], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_818, primals_818, 1, grid=grid(1), stream=stream0)
        del primals_818
        # Source Nodes: [sp_257], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_821, primals_821, 1, grid=grid(1), stream=stream0)
        del primals_821
        # Source Nodes: [out_157], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_824, primals_824, 1, grid=grid(1), stream=stream0)
        del primals_824
        # Source Nodes: [out_161], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_827, primals_827, 1, grid=grid(1), stream=stream0)
        del primals_827
        # Source Nodes: [sp_262], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_830, primals_830, 1, grid=grid(1), stream=stream0)
        del primals_830
        # Source Nodes: [sp_266], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_833, primals_833, 1, grid=grid(1), stream=stream0)
        del primals_833
        # Source Nodes: [sp_270], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_836, primals_836, 1, grid=grid(1), stream=stream0)
        del primals_836
        # Source Nodes: [out_165], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_839, primals_839, 1, grid=grid(1), stream=stream0)
        del primals_839
        # Source Nodes: [out_169], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_842, primals_842, 1, grid=grid(1), stream=stream0)
        del primals_842
        # Source Nodes: [sp_275], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_845, primals_845, 1, grid=grid(1), stream=stream0)
        del primals_845
        # Source Nodes: [sp_279], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_848, primals_848, 1, grid=grid(1), stream=stream0)
        del primals_848
        # Source Nodes: [sp_283], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_851, primals_851, 1, grid=grid(1), stream=stream0)
        del primals_851
        # Source Nodes: [out_173], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_854, primals_854, 1, grid=grid(1), stream=stream0)
        del primals_854
        # Source Nodes: [out_177], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_857, primals_857, 1, grid=grid(1), stream=stream0)
        del primals_857
        # Source Nodes: [sp_288], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_860, primals_860, 1, grid=grid(1), stream=stream0)
        del primals_860
        # Source Nodes: [sp_292], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_863, primals_863, 1, grid=grid(1), stream=stream0)
        del primals_863
        # Source Nodes: [sp_296], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_866, primals_866, 1, grid=grid(1), stream=stream0)
        del primals_866
        # Source Nodes: [out_181], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_869, primals_869, 1, grid=grid(1), stream=stream0)
        del primals_869
        # Source Nodes: [out_185], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_872, primals_872, 1, grid=grid(1), stream=stream0)
        del primals_872
        # Source Nodes: [sp_301], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_875, primals_875, 1, grid=grid(1), stream=stream0)
        del primals_875
        # Source Nodes: [sp_305], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_878, primals_878, 1, grid=grid(1), stream=stream0)
        del primals_878
        # Source Nodes: [sp_309], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_881, primals_881, 1, grid=grid(1), stream=stream0)
        del primals_881
        # Source Nodes: [out_189], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_884, primals_884, 1, grid=grid(1), stream=stream0)
        del primals_884
        # Source Nodes: [out_193], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_887, primals_887, 1, grid=grid(1), stream=stream0)
        del primals_887
        # Source Nodes: [sp_314], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_890, primals_890, 1, grid=grid(1), stream=stream0)
        del primals_890
        # Source Nodes: [sp_318], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_893, primals_893, 1, grid=grid(1), stream=stream0)
        del primals_893
        # Source Nodes: [sp_322], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_896, primals_896, 1, grid=grid(1), stream=stream0)
        del primals_896
        # Source Nodes: [out_197], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_899, primals_899, 1, grid=grid(1), stream=stream0)
        del primals_899
        # Source Nodes: [out_201], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_902, primals_902, 1, grid=grid(1), stream=stream0)
        del primals_902
        # Source Nodes: [sp_327], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_905, primals_905, 1, grid=grid(1), stream=stream0)
        del primals_905
        # Source Nodes: [sp_331], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_908, primals_908, 1, grid=grid(1), stream=stream0)
        del primals_908
        # Source Nodes: [sp_335], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_911, primals_911, 1, grid=grid(1), stream=stream0)
        del primals_911
        # Source Nodes: [out_205], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_914, primals_914, 1, grid=grid(1), stream=stream0)
        del primals_914
        # Source Nodes: [out_209], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_917, primals_917, 1, grid=grid(1), stream=stream0)
        del primals_917
        # Source Nodes: [sp_340], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_920, primals_920, 1, grid=grid(1), stream=stream0)
        del primals_920
        # Source Nodes: [sp_344], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_923, primals_923, 1, grid=grid(1), stream=stream0)
        del primals_923
        # Source Nodes: [sp_348], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_926, primals_926, 1, grid=grid(1), stream=stream0)
        del primals_926
        # Source Nodes: [out_213], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_929, primals_929, 1, grid=grid(1), stream=stream0)
        del primals_929
        # Source Nodes: [out_217], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_932, primals_932, 1, grid=grid(1), stream=stream0)
        del primals_932
        # Source Nodes: [sp_353], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_935, primals_935, 1, grid=grid(1), stream=stream0)
        del primals_935
        # Source Nodes: [sp_357], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_938, primals_938, 1, grid=grid(1), stream=stream0)
        del primals_938
        # Source Nodes: [sp_361], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_941, primals_941, 1, grid=grid(1), stream=stream0)
        del primals_941
        # Source Nodes: [out_221], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_944, primals_944, 1, grid=grid(1), stream=stream0)
        del primals_944
        # Source Nodes: [out_225], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_947, primals_947, 1, grid=grid(1), stream=stream0)
        del primals_947
        # Source Nodes: [sp_366], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_950, primals_950, 1, grid=grid(1), stream=stream0)
        del primals_950
        # Source Nodes: [sp_370], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_953, primals_953, 1, grid=grid(1), stream=stream0)
        del primals_953
        # Source Nodes: [sp_374], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_956, primals_956, 1, grid=grid(1), stream=stream0)
        del primals_956
        # Source Nodes: [out_229], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_959, primals_959, 1, grid=grid(1), stream=stream0)
        del primals_959
        # Source Nodes: [out_233], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_962, primals_962, 1, grid=grid(1), stream=stream0)
        del primals_962
        # Source Nodes: [sp_379], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_965, primals_965, 1, grid=grid(1), stream=stream0)
        del primals_965
        # Source Nodes: [sp_383], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_968, primals_968, 1, grid=grid(1), stream=stream0)
        del primals_968
        # Source Nodes: [sp_387], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_971, primals_971, 1, grid=grid(1), stream=stream0)
        del primals_971
        # Source Nodes: [out_237], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_974, primals_974, 1, grid=grid(1), stream=stream0)
        del primals_974
        # Source Nodes: [out_241], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_977, primals_977, 1, grid=grid(1), stream=stream0)
        del primals_977
        # Source Nodes: [sp_392], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_980, primals_980, 1, grid=grid(1), stream=stream0)
        del primals_980
        # Source Nodes: [sp_396], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_983, primals_983, 1, grid=grid(1), stream=stream0)
        del primals_983
        # Source Nodes: [sp_400], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_986, primals_986, 1, grid=grid(1), stream=stream0)
        del primals_986
        # Source Nodes: [out_245], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_989, primals_989, 1, grid=grid(1), stream=stream0)
        del primals_989
        # Source Nodes: [shortcut_34], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_992, primals_992, 1, grid=grid(1), stream=stream0)
        del primals_992
        # Source Nodes: [out_249], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_995, primals_995, 1, grid=grid(1), stream=stream0)
        del primals_995
        # Source Nodes: [sp_405], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_998, primals_998, 1, grid=grid(1), stream=stream0)
        del primals_998
        # Source Nodes: [sp_409], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_1001, primals_1001, 1, grid=grid(1), stream=stream0)
        del primals_1001
        # Source Nodes: [sp_413], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_1004, primals_1004, 1, grid=grid(1), stream=stream0)
        del primals_1004
        # Source Nodes: [out_253], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_1007, primals_1007, 1, grid=grid(1), stream=stream0)
        del primals_1007
        # Source Nodes: [out_257], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_1010, primals_1010, 1, grid=grid(1), stream=stream0)
        del primals_1010
        # Source Nodes: [sp_418], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_1013, primals_1013, 1, grid=grid(1), stream=stream0)
        del primals_1013
        # Source Nodes: [sp_422], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_1016, primals_1016, 1, grid=grid(1), stream=stream0)
        del primals_1016
        # Source Nodes: [sp_426], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_1019, primals_1019, 1, grid=grid(1), stream=stream0)
        del primals_1019
        # Source Nodes: [out_261], Original ATen: [aten.add]
        triton_poi_fused_add_98.run(primals_1022, primals_1022, 1, grid=grid(1), stream=stream0)
        del primals_1022
        return (buf2016, buf0, primals_2, primals_4, primals_5, buf1, primals_8, buf2, primals_11, buf3, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, buf4, primals_26, buf5, primals_29, buf6, primals_32, primals_34, primals_35, primals_37, primals_38, buf7, primals_41, buf8, primals_44, buf9, primals_47, primals_49, primals_50, primals_52, primals_53, buf10, primals_56, buf11, primals_59, buf12, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, buf13, primals_74, buf14, primals_77, buf15, primals_80, primals_82, primals_83, primals_85, primals_86, buf16, primals_89, buf17, primals_92, buf18, primals_95, primals_97, primals_98, primals_100, primals_101, buf19, primals_104, buf20, primals_107, buf21, primals_110, primals_112, primals_113, primals_115, primals_116, buf22, primals_119, buf23, primals_122, buf24, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, buf25, primals_137, buf26, primals_140, buf27, primals_143, primals_145, primals_146, primals_148, primals_149, buf28, primals_152, buf29, primals_155, buf30, primals_158, primals_160, primals_161, primals_163, primals_164, buf31, primals_167, buf32, primals_170, buf33, primals_173, primals_175, primals_176, primals_178, primals_179, buf34, primals_182, buf35, primals_185, buf36, primals_188, primals_190, primals_191, primals_193, primals_194, buf37, primals_197, buf38, primals_200, buf39, primals_203, primals_205, primals_206, primals_208, primals_209, buf40, primals_212, buf41, primals_215, buf42, primals_218, primals_220, primals_221, primals_223, primals_224, buf43, primals_227, buf44, primals_230, buf45, primals_233, primals_235, primals_236, primals_238, primals_239, buf46, primals_242, buf47, primals_245, buf48, primals_248, primals_250, primals_251, primals_253, primals_254, buf49, primals_257, buf50, primals_260, buf51, primals_263, primals_265, primals_266, primals_268, primals_269, buf52, primals_272, buf53, primals_275, buf54, primals_278, primals_280, primals_281, primals_283, primals_284, buf55, primals_287, buf56, primals_290, buf57, primals_293, primals_295, primals_296, primals_298, primals_299, buf58, primals_302, buf59, primals_305, buf60, primals_308, primals_310, primals_311, primals_313, primals_314, buf61, primals_317, buf62, primals_320, buf63, primals_323, primals_325, primals_326, primals_328, primals_329, buf64, primals_332, buf65, primals_335, buf66, primals_338, primals_340, primals_341, primals_343, primals_344, buf67, primals_347, buf68, primals_350, buf69, primals_353, primals_355, primals_356, primals_358, primals_359, buf70, primals_362, buf71, primals_365, buf72, primals_368, primals_370, primals_371, primals_373, primals_374, buf73, primals_377, buf74, primals_380, buf75, primals_383, primals_385, primals_386, primals_388, primals_389, buf76, primals_392, buf77, primals_395, buf78, primals_398, primals_400, primals_401, primals_403, primals_404, buf79, primals_407, buf80, primals_410, buf81, primals_413, primals_415, primals_416, primals_418, primals_419, buf82, primals_422, buf83, primals_425, buf84, primals_428, primals_430, primals_431, primals_433, primals_434, buf85, primals_437, buf86, primals_440, buf87, primals_443, primals_445, primals_446, primals_448, primals_449, buf88, primals_452, buf89, primals_455, buf90, primals_458, primals_460, primals_461, primals_463, primals_464, buf91, primals_467, buf92, primals_470, buf93, primals_473, primals_475, primals_476, primals_478, primals_479, primals_481, primals_482, buf94, primals_485, buf95, primals_488, buf96, primals_491, primals_493, primals_494, primals_496, primals_497, buf97, primals_500, buf98, primals_503, buf99, primals_506, primals_508, primals_509, buf100, buf102, buf112, buf113, buf114, buf115, buf117, buf127, reinterpret_tensor(buf128, (8, 26, 56, 56), (326144, 1, 5824, 104), 0), buf130, buf140, reinterpret_tensor(buf128, (8, 26, 56, 56), (326144, 1, 5824, 104), 26), buf143, buf153, reinterpret_tensor(buf128, (8, 26, 56, 56), (326144, 1, 5824, 104), 52), buf156, buf166, reinterpret_tensor(buf128, (8, 26, 56, 56), (326144, 1, 5824, 104), 78), buf170, buf172, buf182, buf184, buf194, buf196, buf198, buf208, reinterpret_tensor(buf209, (8, 26, 56, 56), (326144, 1, 5824, 104), 0), buf211, buf221, buf223, buf225, buf235, buf237, buf239, buf249, buf253, buf255, buf265, buf266, buf268, buf278, reinterpret_tensor(buf279, (8, 26, 56, 56), (326144, 1, 5824, 104), 0), buf281, buf291, buf293, buf295, buf305, buf307, buf309, buf319, buf323, buf325, buf335, buf336, buf338, buf348, reinterpret_tensor(buf349, (8, 52, 56, 56), (652288, 1, 11648, 208), 0), buf351, buf358, reinterpret_tensor(buf349, (8, 52, 56, 56), (652288, 1, 11648, 208), 52), buf361, buf368, reinterpret_tensor(buf349, (8, 52, 56, 56), (652288, 1, 11648, 208), 104), buf371, buf378, reinterpret_tensor(buf349, (8, 52, 56, 56), (652288, 1, 11648, 208), 156), buf382, buf384, buf391, buf393, buf400, buf402, buf404, buf411, reinterpret_tensor(buf412, (8, 52, 28, 28), (163072, 1, 5824, 208), 0), buf414, buf421, buf423, buf425, buf432, buf434, buf436, buf443, buf447, buf449, buf456, buf457, buf459, buf466, reinterpret_tensor(buf467, (8, 52, 28, 28), (163072, 1, 5824, 208), 0), buf469, buf476, buf478, buf480, buf487, buf489, buf491, buf498, buf502, buf504, buf511, buf512, buf514, buf521, reinterpret_tensor(buf522, (8, 52, 28, 28), (163072, 1, 5824, 208), 0), buf524, buf531, buf533, buf535, buf542, buf544, buf546, buf553, buf557, buf559, buf566, buf567, buf569, buf576, reinterpret_tensor(buf577, (8, 104, 28, 28), (326144, 1, 11648, 416), 0), buf579, buf586, reinterpret_tensor(buf577, (8, 104, 28, 28), (326144, 1, 11648, 416), 104), buf589, buf596, reinterpret_tensor(buf577, (8, 104, 28, 28), (326144, 1, 11648, 416), 208), buf599, buf606, reinterpret_tensor(buf577, (8, 104, 28, 28), (326144, 1, 11648, 416), 312), buf610, buf612, buf619, buf621, buf628, buf630, buf632, buf639, reinterpret_tensor(buf640, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf642, buf649, buf651, buf653, buf660, buf662, buf664, buf671, buf675, buf677, buf684, buf685, buf687, buf694, reinterpret_tensor(buf695, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf697, buf704, buf706, buf708, buf715, buf717, buf719, buf726, buf730, buf732, buf739, buf740, buf742, buf749, reinterpret_tensor(buf750, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf752, buf759, buf761, buf763, buf770, buf772, buf774, buf781, buf785, buf787, buf794, buf795, buf797, buf804, reinterpret_tensor(buf805, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf807, buf814, buf816, buf818, buf825, buf827, buf829, buf836, buf840, buf842, buf849, buf850, buf852, buf859, reinterpret_tensor(buf860, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf862, buf869, buf871, buf873, buf880, buf882, buf884, buf891, buf895, buf897, buf904, buf905, buf907, buf914, reinterpret_tensor(buf915, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf917, buf924, buf926, buf928, buf935, buf937, buf939, buf946, buf950, buf952, buf959, buf960, buf962, buf969, reinterpret_tensor(buf970, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf972, buf979, buf981, buf983, buf990, buf992, buf994, buf1001, buf1005, buf1007, buf1014, buf1015, buf1017, buf1024, reinterpret_tensor(buf1025, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf1027, buf1034, buf1036, buf1038, buf1045, buf1047, buf1049, buf1056, buf1060, buf1062, buf1069, buf1070, buf1072, buf1079, reinterpret_tensor(buf1080, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf1082, buf1089, buf1091, buf1093, buf1100, buf1102, buf1104, buf1111, buf1115, buf1117, buf1124, buf1125, buf1127, buf1134, reinterpret_tensor(buf1135, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf1137, buf1144, buf1146, buf1148, buf1155, buf1157, buf1159, buf1166, buf1170, buf1172, buf1179, buf1180, buf1182, buf1189, reinterpret_tensor(buf1190, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf1192, buf1199, buf1201, buf1203, buf1210, buf1212, buf1214, buf1221, buf1225, buf1227, buf1234, buf1235, buf1237, buf1244, reinterpret_tensor(buf1245, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf1247, buf1254, buf1256, buf1258, buf1265, buf1267, buf1269, buf1276, buf1280, buf1282, buf1289, buf1290, buf1292, buf1299, reinterpret_tensor(buf1300, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf1302, buf1309, buf1311, buf1313, buf1320, buf1322, buf1324, buf1331, buf1335, buf1337, buf1344, buf1345, buf1347, buf1354, reinterpret_tensor(buf1355, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf1357, buf1364, buf1366, buf1368, buf1375, buf1377, buf1379, buf1386, buf1390, buf1392, buf1399, buf1400, buf1402, buf1409, reinterpret_tensor(buf1410, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf1412, buf1419, buf1421, buf1423, buf1430, buf1432, buf1434, buf1441, buf1445, buf1447, buf1454, buf1455, buf1457, buf1464, reinterpret_tensor(buf1465, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf1467, buf1474, buf1476, buf1478, buf1485, buf1487, buf1489, buf1496, buf1500, buf1502, buf1509, buf1510, buf1512, buf1519, reinterpret_tensor(buf1520, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf1522, buf1529, buf1531, buf1533, buf1540, buf1542, buf1544, buf1551, buf1555, buf1557, buf1564, buf1565, buf1567, buf1574, reinterpret_tensor(buf1575, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf1577, buf1584, buf1586, buf1588, buf1595, buf1597, buf1599, buf1606, buf1610, buf1612, buf1619, buf1620, buf1622, buf1629, reinterpret_tensor(buf1630, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf1632, buf1639, buf1641, buf1643, buf1650, buf1652, buf1654, buf1661, buf1665, buf1667, buf1674, buf1675, buf1677, buf1684, reinterpret_tensor(buf1685, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf1687, buf1694, buf1696, buf1698, buf1705, buf1707, buf1709, buf1716, buf1720, buf1722, buf1729, buf1730, buf1732, buf1739, reinterpret_tensor(buf1740, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf1742, buf1749, buf1751, buf1753, buf1760, buf1762, buf1764, buf1771, buf1775, buf1777, buf1784, buf1785, buf1787, buf1794, reinterpret_tensor(buf1795, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf1797, buf1804, buf1806, buf1808, buf1815, buf1817, buf1819, buf1826, buf1830, buf1832, buf1839, buf1840, buf1842, buf1849, reinterpret_tensor(buf1850, (8, 208, 14, 14), (163072, 1, 11648, 832), 0), buf1852, buf1859, reinterpret_tensor(buf1850, (8, 208, 14, 14), (163072, 1, 11648, 832), 208), buf1862, buf1869, reinterpret_tensor(buf1850, (8, 208, 14, 14), (163072, 1, 11648, 832), 416), buf1872, buf1879, reinterpret_tensor(buf1850, (8, 208, 14, 14), (163072, 1, 11648, 832), 624), buf1883, buf1885, buf1892, buf1894, buf1901, buf1903, buf1905, buf1912, reinterpret_tensor(buf1913, (8, 208, 7, 7), (40768, 1, 5824, 832), 0), buf1915, buf1922, buf1924, buf1926, buf1933, buf1935, buf1937, buf1944, buf1948, buf1950, buf1957, buf1958, buf1960, buf1967, reinterpret_tensor(buf1968, (8, 208, 7, 7), (40768, 1, 5824, 832), 0), buf1970, buf1977, buf1979, buf1981, buf1988, buf1990, buf1992, buf1999, buf2003, buf2005, buf2012, buf2015, reinterpret_tensor(primals_511, (1000, 2048), (2048, 1), 0), buf2017, reinterpret_tensor(buf2009, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), buf2018, reinterpret_tensor(buf1996, (1, 208, 1, 1), (208, 1, 1, 1), 0), buf2019, reinterpret_tensor(buf1985, (1, 208, 1, 1), (208, 1, 1, 1), 0), buf2020, reinterpret_tensor(buf1974, (1, 208, 1, 1), (208, 1, 1, 1), 0), buf2021, reinterpret_tensor(buf1964, (1, 832, 1, 1), (832, 1, 1, 1), 0), reinterpret_tensor(buf1954, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), buf2022, reinterpret_tensor(buf1941, (1, 208, 1, 1), (208, 1, 1, 1), 0), buf2023, reinterpret_tensor(buf1930, (1, 208, 1, 1), (208, 1, 1, 1), 0), buf2024, reinterpret_tensor(buf1919, (1, 208, 1, 1), (208, 1, 1, 1), 0), buf2025, reinterpret_tensor(buf1909, (1, 832, 1, 1), (832, 1, 1, 1), 0), reinterpret_tensor(buf1898, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf1889, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), buf2026, reinterpret_tensor(buf1876, (1, 208, 1, 1), (208, 1, 1, 1), 0), buf2027, reinterpret_tensor(buf1866, (1, 208, 1, 1), (208, 1, 1, 1), 0), buf2028, reinterpret_tensor(buf1856, (1, 208, 1, 1), (208, 1, 1, 1), 0), buf2029, reinterpret_tensor(buf1846, (1, 832, 1, 1), (832, 1, 1, 1), 0), reinterpret_tensor(buf1836, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2030, reinterpret_tensor(buf1823, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2031, reinterpret_tensor(buf1812, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2032, reinterpret_tensor(buf1801, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2033, reinterpret_tensor(buf1791, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf1781, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2034, reinterpret_tensor(buf1768, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2035, reinterpret_tensor(buf1757, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2036, reinterpret_tensor(buf1746, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2037, reinterpret_tensor(buf1736, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf1726, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2038, reinterpret_tensor(buf1713, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2039, reinterpret_tensor(buf1702, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2040, reinterpret_tensor(buf1691, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2041, reinterpret_tensor(buf1681, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf1671, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2042, reinterpret_tensor(buf1658, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2043, reinterpret_tensor(buf1647, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2044, reinterpret_tensor(buf1636, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2045, reinterpret_tensor(buf1626, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf1616, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2046, reinterpret_tensor(buf1603, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2047, reinterpret_tensor(buf1592, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2048, reinterpret_tensor(buf1581, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2049, reinterpret_tensor(buf1571, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf1561, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2050, reinterpret_tensor(buf1548, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2051, reinterpret_tensor(buf1537, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2052, reinterpret_tensor(buf1526, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2053, reinterpret_tensor(buf1516, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf1506, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2054, reinterpret_tensor(buf1493, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2055, reinterpret_tensor(buf1482, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2056, reinterpret_tensor(buf1471, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2057, reinterpret_tensor(buf1461, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf1451, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2058, reinterpret_tensor(buf1438, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2059, reinterpret_tensor(buf1427, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2060, reinterpret_tensor(buf1416, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2061, reinterpret_tensor(buf1406, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf1396, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2062, reinterpret_tensor(buf1383, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2063, reinterpret_tensor(buf1372, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2064, reinterpret_tensor(buf1361, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2065, reinterpret_tensor(buf1351, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf1341, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2066, reinterpret_tensor(buf1328, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2067, reinterpret_tensor(buf1317, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2068, reinterpret_tensor(buf1306, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2069, reinterpret_tensor(buf1296, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf1286, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2070, reinterpret_tensor(buf1273, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2071, reinterpret_tensor(buf1262, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2072, reinterpret_tensor(buf1251, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2073, reinterpret_tensor(buf1241, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf1231, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2074, reinterpret_tensor(buf1218, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2075, reinterpret_tensor(buf1207, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2076, reinterpret_tensor(buf1196, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2077, reinterpret_tensor(buf1186, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf1176, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2078, reinterpret_tensor(buf1163, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2079, reinterpret_tensor(buf1152, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2080, reinterpret_tensor(buf1141, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2081, reinterpret_tensor(buf1131, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf1121, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2082, reinterpret_tensor(buf1108, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2083, reinterpret_tensor(buf1097, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2084, reinterpret_tensor(buf1086, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2085, reinterpret_tensor(buf1076, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf1066, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2086, reinterpret_tensor(buf1053, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2087, reinterpret_tensor(buf1042, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2088, reinterpret_tensor(buf1031, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2089, reinterpret_tensor(buf1021, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf1011, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2090, reinterpret_tensor(buf998, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2091, reinterpret_tensor(buf987, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2092, reinterpret_tensor(buf976, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2093, reinterpret_tensor(buf966, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf956, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2094, reinterpret_tensor(buf943, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2095, reinterpret_tensor(buf932, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2096, reinterpret_tensor(buf921, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2097, reinterpret_tensor(buf911, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf901, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2098, reinterpret_tensor(buf888, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2099, reinterpret_tensor(buf877, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2100, reinterpret_tensor(buf866, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2101, reinterpret_tensor(buf856, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf846, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2102, reinterpret_tensor(buf833, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2103, reinterpret_tensor(buf822, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2104, reinterpret_tensor(buf811, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2105, reinterpret_tensor(buf801, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf791, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2106, reinterpret_tensor(buf778, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2107, reinterpret_tensor(buf767, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2108, reinterpret_tensor(buf756, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2109, reinterpret_tensor(buf746, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf736, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2110, reinterpret_tensor(buf723, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2111, reinterpret_tensor(buf712, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2112, reinterpret_tensor(buf701, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2113, reinterpret_tensor(buf691, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf681, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2114, reinterpret_tensor(buf668, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2115, reinterpret_tensor(buf657, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2116, reinterpret_tensor(buf646, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2117, reinterpret_tensor(buf636, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf625, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf616, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf2118, reinterpret_tensor(buf603, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2119, reinterpret_tensor(buf593, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2120, reinterpret_tensor(buf583, (1, 104, 1, 1), (104, 1, 1, 1), 0), buf2121, reinterpret_tensor(buf573, (1, 416, 1, 1), (416, 1, 1, 1), 0), reinterpret_tensor(buf563, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf2122, reinterpret_tensor(buf550, (1, 52, 1, 1), (52, 1, 1, 1), 0), buf2123, reinterpret_tensor(buf539, (1, 52, 1, 1), (52, 1, 1, 1), 0), buf2124, reinterpret_tensor(buf528, (1, 52, 1, 1), (52, 1, 1, 1), 0), buf2125, reinterpret_tensor(buf518, (1, 208, 1, 1), (208, 1, 1, 1), 0), reinterpret_tensor(buf508, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf2126, reinterpret_tensor(buf495, (1, 52, 1, 1), (52, 1, 1, 1), 0), buf2127, reinterpret_tensor(buf484, (1, 52, 1, 1), (52, 1, 1, 1), 0), buf2128, reinterpret_tensor(buf473, (1, 52, 1, 1), (52, 1, 1, 1), 0), buf2129, reinterpret_tensor(buf463, (1, 208, 1, 1), (208, 1, 1, 1), 0), reinterpret_tensor(buf453, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf2130, reinterpret_tensor(buf440, (1, 52, 1, 1), (52, 1, 1, 1), 0), buf2131, reinterpret_tensor(buf429, (1, 52, 1, 1), (52, 1, 1, 1), 0), buf2132, reinterpret_tensor(buf418, (1, 52, 1, 1), (52, 1, 1, 1), 0), buf2133, reinterpret_tensor(buf408, (1, 208, 1, 1), (208, 1, 1, 1), 0), reinterpret_tensor(buf397, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf388, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf2134, reinterpret_tensor(buf375, (1, 52, 1, 1), (52, 1, 1, 1), 0), buf2135, reinterpret_tensor(buf365, (1, 52, 1, 1), (52, 1, 1, 1), 0), buf2136, reinterpret_tensor(buf355, (1, 52, 1, 1), (52, 1, 1, 1), 0), buf2137, reinterpret_tensor(buf345, (1, 208, 1, 1), (208, 1, 1, 1), 0), reinterpret_tensor(buf332, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf2138, reinterpret_tensor(buf316, (1, 26, 1, 1), (26, 1, 1, 1), 0), buf2139, reinterpret_tensor(buf302, (1, 26, 1, 1), (26, 1, 1, 1), 0), buf2140, reinterpret_tensor(buf288, (1, 26, 1, 1), (26, 1, 1, 1), 0), buf2141, reinterpret_tensor(buf275, (1, 104, 1, 1), (104, 1, 1, 1), 0), reinterpret_tensor(buf262, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf2142, reinterpret_tensor(buf246, (1, 26, 1, 1), (26, 1, 1, 1), 0), buf2143, reinterpret_tensor(buf232, (1, 26, 1, 1), (26, 1, 1, 1), 0), buf2144, reinterpret_tensor(buf218, (1, 26, 1, 1), (26, 1, 1, 1), 0), buf2145, reinterpret_tensor(buf205, (1, 104, 1, 1), (104, 1, 1, 1), 0), reinterpret_tensor(buf191, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf179, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf2146, reinterpret_tensor(buf163, (1, 26, 1, 1), (26, 1, 1, 1), 0), buf2147, reinterpret_tensor(buf150, (1, 26, 1, 1), (26, 1, 1, 1), 0), buf2148, reinterpret_tensor(buf137, (1, 26, 1, 1), (26, 1, 1, 1), 0), buf2149, reinterpret_tensor(buf124, (1, 104, 1, 1), (104, 1, 1, 1), 0), reinterpret_tensor(buf109, (1, 64, 1, 1), (64, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((104, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((104, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((104, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((208, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((208, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((208, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((208, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((416, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((832, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((2048, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((832, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((2048, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((832, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((2048, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_516 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_519 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_522 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_525 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_528 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_531 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_534 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_537 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_540 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_543 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_546 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_549 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_552 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_555 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_558 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_561 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_564 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_567 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_570 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_573 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_576 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_579 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_582 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_585 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_588 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_591 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_594 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_597 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_600 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_603 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_606 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_609 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_612 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_615 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_618 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_621 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_624 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_627 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_630 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_633 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_636 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_639 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_642 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_645 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_648 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_651 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_654 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_657 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_660 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_663 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_666 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_669 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_672 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_675 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_678 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_681 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_684 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_687 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_689 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_690 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_692 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_693 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_695 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_696 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_698 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_699 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_701 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_702 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_704 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_705 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_707 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_708 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_710 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_711 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_712 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_713 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_714 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_715 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_716 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_717 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_718 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_719 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_720 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_721 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_722 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_723 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_724 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_725 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_726 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_727 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_728 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_729 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_730 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_731 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_732 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_733 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_734 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_735 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_736 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_737 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_738 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_739 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_740 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_741 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_742 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_743 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_744 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_745 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_746 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_747 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_748 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_749 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_750 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_751 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_752 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_753 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_754 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_755 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_756 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_757 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_758 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_759 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_760 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_761 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_762 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_763 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_764 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_765 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_766 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_767 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_768 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_769 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_770 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_771 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_772 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_773 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_774 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_775 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_776 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_777 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_778 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_779 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_780 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_781 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_782 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_783 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_784 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_785 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_786 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_787 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_788 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_789 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_790 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_791 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_792 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_793 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_794 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_795 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_796 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_797 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_798 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_799 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_800 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_801 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_802 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_803 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_804 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_805 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_806 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_807 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_808 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_809 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_810 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_811 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_812 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_813 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_814 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_815 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_816 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_817 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_818 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_819 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_820 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_821 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_822 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_823 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_824 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_825 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_826 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_827 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_828 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_829 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_830 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_831 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_832 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_833 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_834 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_835 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_836 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_837 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_838 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_839 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_840 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_841 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_842 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_843 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_844 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_845 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_846 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_847 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_848 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_849 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_850 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_851 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_852 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_853 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_854 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_855 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_856 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_857 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_858 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_859 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_860 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_861 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_862 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_863 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_864 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_865 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_866 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_867 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_868 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_869 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_870 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_871 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_872 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_873 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_874 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_875 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_876 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_877 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_878 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_879 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_880 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_881 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_882 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_883 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_884 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_885 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_886 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_887 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_888 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_889 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_890 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_891 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_892 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_893 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_894 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_895 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_896 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_897 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_898 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_899 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_900 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_901 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_902 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_903 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_904 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_905 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_906 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_907 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_908 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_909 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_910 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_911 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_912 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_913 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_914 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_915 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_916 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_917 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_918 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_919 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_920 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_921 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_922 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_923 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_924 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_925 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_926 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_927 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_928 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_929 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_930 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_931 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_932 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_933 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_934 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_935 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_936 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_937 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_938 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_939 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_940 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_941 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_942 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_943 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_944 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_945 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_946 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_947 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_948 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_949 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_950 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_951 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_952 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_953 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_954 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_955 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_956 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_957 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_958 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_959 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_960 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_961 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_962 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_963 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_964 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_965 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_966 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_967 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_968 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_969 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_970 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_971 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_972 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_973 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_974 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_975 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_976 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_977 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_978 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_979 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_980 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_981 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_982 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_983 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_984 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_985 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_986 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_987 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_988 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_989 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_990 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_991 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_992 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_993 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_994 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_995 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_996 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_997 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_998 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_999 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1000 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1001 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_1002 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1003 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1004 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_1005 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1006 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1007 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_1008 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1009 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1010 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_1011 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1012 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1013 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_1014 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1015 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1016 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_1017 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1018 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1019 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_1020 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1021 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1022 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_1023 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_945, primals_946, primals_947, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_957, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_964, primals_965, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, primals_977, primals_978, primals_979, primals_980, primals_981, primals_982, primals_983, primals_984, primals_985, primals_986, primals_987, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_995, primals_996, primals_997, primals_998, primals_999, primals_1000, primals_1001, primals_1002, primals_1003, primals_1004, primals_1005, primals_1006, primals_1007, primals_1008, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1014, primals_1015, primals_1016, primals_1017, primals_1018, primals_1019, primals_1020, primals_1021, primals_1022, primals_1023]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('res2net101_26w_4s', benchmark_compiled_module)
