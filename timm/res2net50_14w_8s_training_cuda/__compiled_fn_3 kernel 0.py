
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


# kernel path: /tmp/torchinductor_youkaichao/st/cstf5bzhvtzqojs6ctycqebrsaw5tf3er2cldxtrbeqawn6pxbsx.py
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
    size_hints=[256, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (14*x2) + (126*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7d/c7dgtdvpgfbdscvtbdsh5onvtomcuu77s6dorj77pw6lsnjv4kx6.py
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
    size_hints=[1024, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (28*x2) + (252*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yf/cyf57ddpvxrmuxjmix2vuyslah6yhiprvzxix4hbfqnuqddh6qmv.py
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
    size_hints=[4096, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (56*x2) + (504*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vr/cvrrydxts6cv7pj76ilf2w373euzqolfj7jtp3se2si3nmeq5lz6.py
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
    size_hints=[16384, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (1008*y1)), tmp0, xmask & ymask)
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


# kernel path: /tmp/torchinductor_youkaichao/kh/ckhpqlb3yj353yl2z64bsras3qvv75zfejuoye7f6wxih7masmif.py
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
    ynumel = 896
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (351232*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vj/cvj2meajihmpdhvjfwkkc5p5edxampqrym77uwagk3basjk2f3nk.py
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
    xnumel = 21952
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 112
    x1 = (xindex // 112)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (112*r2) + (14336*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/gd/cgd6benqsvm2pl63xczsxhujcsqytposobdf4jvdqj7szzefist2.py
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
    xnumel = 224
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
        tmp0 = tl.load(in_ptr0 + (x1 + (112*r2) + (10976*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (112*r2) + (10976*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (112*r2) + (10976*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (112*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (112*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (112*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qf/cqftngta33rnyyaawz5hyh76gvytzb7wprrxqzrhgxdn7bku2rot.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_15', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (112*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (112*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (112*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/kj/ckjv6oou6yxypx7p5vbl45mhb2ziuoerv5key63h4av2wsmgaljg.py
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
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 112
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


# kernel path: /tmp/torchinductor_youkaichao/5s/c5s6crdzm53wwgixpjq2ykkqzy73gyu67d5mmpgqtsaa2emrqtmj.py
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
    size_hints=[128, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (14*x2) + (43904*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3y/c3yvp4msrhbupihruhqjs4dm4bwhcg3i3uira2wr7prcewa5wwyh.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2744
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 14
    x1 = (xindex // 14)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (14*r2) + (1792*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/vw/cvwfrihk55ije5vdhizidclk4ac56txv65hjbdn3yj4bi3ifqynh.py
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
    size_hints=[32, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 28
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
        tmp0 = tl.load(in_ptr0 + (x1 + (14*r2) + (1372*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (14*r2) + (1372*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (14*r2) + (1372*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (14*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (14*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (14*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qt/cqtb4yxbbaaubcdedsxgzle5q6p5nj23j4vn5r32icqjitpcnc4z.py
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
    size_hints=[16, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 14
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (14*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (14*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (14*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/te/cte7z2ccfcwep4abdbt62ho5hq4kvx44vm3yaftnfvnxrk6q632y.py
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
    size_hints=[128, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (y0 + (14*x2) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (3136*y0) + (351232*y1)), tmp14, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (14*x2) + (43904*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2g/c2ghxmiudx25se2tqulpq6six657thwuuq7nh3zegm2ogdbkploo.py
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
    size_hints=[128, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
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
    y0 = yindex % 14
    y1 = (yindex // 14)
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
    tmp11 = tl.load(in_ptr0 + ((-6286) + y0 + (112*x5) + (351232*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x2
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-6174) + y0 + (112*x5) + (351232*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x2
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-6062) + y0 + (112*x5) + (351232*y1)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x3
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-14) + y0 + (112*x5) + (351232*y1)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (98 + y0 + (112*x5) + (351232*y1)), tmp41 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (210 + y0 + (112*x5) + (351232*y1)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x3
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (6258 + y0 + (112*x5) + (351232*y1)), tmp55 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (6370 + y0 + (112*x5) + (351232*y1)), tmp60 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (6482 + y0 + (112*x5) + (351232*y1)), tmp65 & xmask & ymask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x5 + (3136*y0) + (351232*y1)), tmp178, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7j/c7jajy2iao6jhxbvsdcotaylopztqc27uby3xgqewdmciqzizx34.py
# Source Nodes: [out_4], Original ATen: [aten.convolution]
# out_4 => convolution_9
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
# out_5 => var_mean_9
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
# out_5 => var_mean_9
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
# out_5 => add_46, add_47, add_48, mul_64, mul_65, mul_66, mul_67, mul_68, rsqrt_9, squeeze_28, var_mean_9
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
# out_5 => add_46, add_49, mul_63, mul_69, rsqrt_9, sub_9, var_mean_9
# out_6 => add_55
# shortcut_1 => add_51, add_54, mul_70, mul_76, rsqrt_10, sub_10, var_mean_10
# shortcut_2 => relu_9
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


# kernel path: /tmp/torchinductor_youkaichao/sc/cscbxnjohl6lbkinwimmooqeuu2qyk3imanqpch3fkdr34hqnt2b.py
# Source Nodes: [sp_33], Original ATen: [aten.add]
# sp_33 => add_66
triton_poi_fused_add_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 14
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
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (14 + x2 + (112*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (14*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gt/cgtm5elik44ql63b5hiy4ibnrc7mnrasggbswub5antqxo2dnwe4.py
# Source Nodes: [sp_37], Original ATen: [aten.add]
# sp_37 => add_72
triton_poi_fused_add_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 14
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
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (28 + x2 + (112*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (14*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2f/c2fzygq7r2r3vybtn47fqmcbhvf6djv7y6dagx765jvi37ewyb3u.py
# Source Nodes: [sp_41], Original ATen: [aten.add]
# sp_41 => add_78
triton_poi_fused_add_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 14
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
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (42 + x2 + (112*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (14*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mv/cmvligxkledkkszucaideqd4gltxml26b2lgv4o2vp7rtutb73p6.py
# Source Nodes: [sp_45], Original ATen: [aten.add]
# sp_45 => add_84
triton_poi_fused_add_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 14
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
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (56 + x2 + (112*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (14*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wl/cwlp3a7j6nzx4zfrwaykvp23gwjkwjtis7ozkaaul63spwzh34ic.py
# Source Nodes: [sp_49], Original ATen: [aten.add]
# sp_49 => add_90
triton_poi_fused_add_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 14
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
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (70 + x2 + (112*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (14*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ms/cmsxmk2rsipjf2bv464ugpc2vgks625upejpuuqqo66h6xfwqygu.py
# Source Nodes: [sp_53], Original ATen: [aten.add]
# sp_53 => add_96
triton_poi_fused_add_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 14
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
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (84 + x2 + (112*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (14*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lh/clhbav4tfc6q7vpoh76fuvlsock3wpk3vmgkd2jpgukicgjfrg2i.py
# Source Nodes: [cat_30], Original ATen: [aten.cat]
# cat_30 => cat_1
triton_poi_fused_cat_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (98 + y0 + (112*x2) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (3136*y0) + (351232*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ot/coty4c7jrf4qknbviicgsjv7wkqg4sqkiwcb3iuazgyxo2tsx4hw.py
# Source Nodes: [out_13, out_14, shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_13 => add_103, add_106, mul_133, mul_139, rsqrt_19, sub_19, var_mean_19
# out_14 => add_107
# shortcut_3 => relu_18
triton_poi_fused__native_batch_norm_legit_functional_add_relu_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_35', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ev/ceve5262osq7ogp24tzvbzheynyoez5jjkbjgzc7hn5k4wr6ag4x.py
# Source Nodes: [out_24], Original ATen: [aten.convolution]
# out_24 => convolution_29
triton_poi_fused_convolution_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1792
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 224
    y1 = (yindex // 224)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (224*x2) + (702464*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/di/cdigbs6qdrg7zri3ml2xfxg6hqwbd4ltvxjj6kgz5hyzgc2dc4nq.py
# Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional]
# out_25 => var_mean_29
triton_red_fused__native_batch_norm_legit_functional_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 43904
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 224
    x1 = (xindex // 224)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (224*r2) + (28672*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/fh/cfhwcezshotgcrj7ize5qlz4bqlpbhvx75uclyw7hi5skxmy3q6u.py
# Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional]
# out_25 => var_mean_29
triton_red_fused__native_batch_norm_legit_functional_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
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
        tmp0 = tl.load(in_ptr0 + (x1 + (224*r2) + (21952*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (224*r2) + (21952*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (224*r2) + (21952*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (224*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (224*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (224*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rd/crd2ahn2umqfkwgixeucrrkwpi7xyb2eq5g2mdzubq4ik3ckzaa4.py
# Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional]
# out_25 => add_161, add_162, add_163, mul_204, mul_205, mul_206, mul_207, mul_208, rsqrt_29, squeeze_88, var_mean_29
triton_per_fused__native_batch_norm_legit_functional_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_39', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (224*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (224*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (224*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/go/cgocupzrx73aosnsxifbbtmfnd7z6vr5pexuwti4qhvetwgvltti.py
# Source Nodes: [out_25, out_26], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_25 => add_161, add_164, mul_203, mul_209, rsqrt_29, sub_29, var_mean_29
# out_26 => relu_28
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5619712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 224
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


# kernel path: /tmp/torchinductor_youkaichao/aj/cajcx4flwnfjhfiy7z6mcsbvk5btw2ip6cmudjnzarcwspdvsj66.py
# Source Nodes: [sp_88], Original ATen: [aten.convolution]
# sp_88 => convolution_30
triton_poi_fused_convolution_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (28*x2) + (21952*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jm/cjmlpkquu6wz2b5nr4vv6cqdx2zybit2sehlz577kji4sit5tpdt.py
# Source Nodes: [sp_89], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_89 => var_mean_30
triton_red_fused__native_batch_norm_legit_functional_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1372
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 28
    x1 = (xindex // 28)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (28*r2) + (3584*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/v4/cv4j5fzgi7uk4cel76wh4ilnoykfqscyhow5nlfou4siws5piyfo.py
# Source Nodes: [sp_89], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_89 => add_166, add_167, add_168, mul_211, mul_212, mul_213, mul_214, mul_215, rsqrt_30, squeeze_91, var_mean_30
triton_per_fused__native_batch_norm_legit_functional_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_43', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 28
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (28*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (28*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (28*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/yf/cyfbcpk3o35oo7rpwzf26ggdpe72u7nu3yhtute6dgmumrm6sdfb.py
# Source Nodes: [sp_89, sp_90], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# sp_89 => add_166, add_169, mul_210, mul_216, rsqrt_30, sub_30, var_mean_30
# sp_90 => relu_29
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (y0 + (28*x2) + (21952*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (784*y0) + (175616*y1)), tmp14, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (28*x2) + (21952*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7t/c7tkgjxd5lt73s4kbo53krqs6l7zsqz57arctgwo7lbizodj75kq.py
# Source Nodes: [getattr_l__mod___layer2___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer2___0___pool => avg_pool2d_1
triton_poi_fused_avg_pool2d_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 28)
    x2 = xindex % 28
    y0 = yindex % 28
    y1 = (yindex // 28)
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
    tmp11 = tl.load(in_ptr0 + ((-12572) + y0 + (448*x2) + (25088*x3) + (702464*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x2
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-12348) + y0 + (448*x2) + (25088*x3) + (702464*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x2)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-12124) + y0 + (448*x2) + (25088*x3) + (702464*y1)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x3
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-28) + y0 + (448*x2) + (25088*x3) + (702464*y1)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (196 + y0 + (448*x2) + (25088*x3) + (702464*y1)), tmp41 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (420 + y0 + (448*x2) + (25088*x3) + (702464*y1)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x3)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (12516 + y0 + (448*x2) + (25088*x3) + (702464*y1)), tmp55 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (12740 + y0 + (448*x2) + (25088*x3) + (702464*y1)), tmp60 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (12964 + y0 + (448*x2) + (25088*x3) + (702464*y1)), tmp65 & xmask & ymask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x5 + (784*y0) + (175616*y1)), tmp178, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tj/ctj2mv7fomfs3mf5aou4sc5cimr4vrnqpjufkphgrhpvzxljj6ra.py
# Source Nodes: [cat_28], Original ATen: [aten.cat]
# cat_28 => cat_3
triton_poi_fused_cat_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1792
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 224
    y1 = (yindex // 224)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (224*x2) + (175616*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gj/cgjkp6oiivm4wg4dj2b5wuusbfc5omcs24oxzmwbirwxhb7ig6it.py
# Source Nodes: [out_28], Original ATen: [aten.convolution]
# out_28 => convolution_37
triton_poi_fused_convolution_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_47', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/x3/cx3a3ygrxmqqdd4vuosfyur65pao2ezonpyaqrjj6p7g7l5zsdp4.py
# Source Nodes: [out_29], Original ATen: [aten._native_batch_norm_legit_functional]
# out_29 => var_mean_37
triton_red_fused__native_batch_norm_legit_functional_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_48', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ga/cgal2hl5xrxnj5t45tpe5apffzoekhezvgflwjyijmoxilc4irbe.py
# Source Nodes: [out_29], Original ATen: [aten._native_batch_norm_legit_functional]
# out_29 => add_201, add_202, add_203, mul_260, mul_261, mul_262, mul_263, mul_264, rsqrt_37, squeeze_112, var_mean_37
triton_per_fused__native_batch_norm_legit_functional_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_49', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/cr/ccraa5brsanzju3v7wgeqr43hf4pojdacl55irvxbr4qogvkvieq.py
# Source Nodes: [out_29, out_30, shortcut_5, shortcut_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_29 => add_201, add_204, mul_259, mul_265, rsqrt_37, sub_37, var_mean_37
# out_30 => add_210
# shortcut_5 => add_206, add_209, mul_266, mul_272, rsqrt_38, sub_38, var_mean_38
# shortcut_6 => relu_36
triton_poi_fused__native_batch_norm_legit_functional_add_relu_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_50', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/rx/crxwcjls4pbfj7ukral6rwuvs4luatqmydw4orbfz7uelfhttlsa.py
# Source Nodes: [out_33], Original ATen: [aten._native_batch_norm_legit_functional]
# out_33 => var_mean_39
triton_red_fused__native_batch_norm_legit_functional_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10976
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 224
    x1 = (xindex // 224)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (224*r2) + (28672*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/tp/ctpelxws67kzhzyirx6g44gxniob7hsjwihecfsgfbkmqkdjwzsg.py
# Source Nodes: [out_33], Original ATen: [aten._native_batch_norm_legit_functional]
# out_33 => add_212, add_213, add_214, mul_274, mul_275, mul_276, mul_277, mul_278, rsqrt_39, squeeze_118, var_mean_39
triton_per_fused__native_batch_norm_legit_functional_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_52', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (224*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (224*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (224*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/t4/ct4tdjatzvhchytywpbjqhncyo2jncbzydb7wc33n3f47hyqylrl.py
# Source Nodes: [out_33, out_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_33 => add_212, add_215, mul_273, mul_279, rsqrt_39, sub_39, var_mean_39
# out_34 => relu_37
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 224
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


# kernel path: /tmp/torchinductor_youkaichao/k2/ck222klvk7z4zjshibcpwm2ggajohb3im5xq7ngz3kzlcsqss3k6.py
# Source Nodes: [sp_120], Original ATen: [aten.add]
# sp_120 => add_221
triton_poi_fused_add_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 28
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (28 + x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (28*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/da/cdacxh27zniqlpu47tb5fe32etcf4ejgrn6wagmn2qm4zzeysb2k.py
# Source Nodes: [sp_124], Original ATen: [aten.add]
# sp_124 => add_227
triton_poi_fused_add_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 28
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (56 + x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (28*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dd/cddkgyznplzkcgql4oqxcayf3afxhy7y7zpd6xjjya6jqrywy62b.py
# Source Nodes: [sp_128], Original ATen: [aten.add]
# sp_128 => add_233
triton_poi_fused_add_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 28
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (84 + x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (28*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b7/cb773k7wp25k7bwv6ecv4lc47kwgjw4ztliyrlvxsxh4f6teyjse.py
# Source Nodes: [sp_132], Original ATen: [aten.add]
# sp_132 => add_239
triton_poi_fused_add_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 28
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (112 + x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (28*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/42/c425t36mwz3c4df6sdwob5whpdrpze5vzr4qju234qkfaumyw6za.py
# Source Nodes: [sp_136], Original ATen: [aten.add]
# sp_136 => add_245
triton_poi_fused_add_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 28
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (140 + x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (28*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rw/crwvae2p7v5f5cepbld2djiupp24v4ntrltjz52e2f3xeytlrxd5.py
# Source Nodes: [sp_140], Original ATen: [aten.add]
# sp_140 => add_251
triton_poi_fused_add_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 28
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (168 + x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (28*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xl/cxlcy4ookafsfx6eatl2tw7vr5q5wqmvuhnzell2uefsf5reuobp.py
# Source Nodes: [cat_27], Original ATen: [aten.cat]
# cat_27 => cat_4
triton_poi_fused_cat_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (196 + y0 + (224*x2) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (784*y0) + (175616*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fq/cfq6kvxdiwnvuio3mwtef4i7by24voxx2gscteuq6vzhjrrsgxrz.py
# Source Nodes: [out_37, out_38, shortcut_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_37 => add_258, add_261, mul_329, mul_335, rsqrt_47, sub_47, var_mean_47
# out_38 => add_262
# shortcut_7 => relu_45
triton_poi_fused__native_batch_norm_legit_functional_add_relu_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_61', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/gd/cgdxzpgizls5b7we437ggnl63czic4c7ccnlnkt3l6kzsjsxtwdz.py
# Source Nodes: [out_56], Original ATen: [aten.convolution]
# out_56 => convolution_66
triton_poi_fused_convolution_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_62', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3584
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 448
    y1 = (yindex // 448)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (448*x2) + (351232*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qv/cqvewamd73zwth26jz5xwwygucqjcnoa2swowrfhi57czvo3bqib.py
# Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional]
# out_57 => var_mean_66
triton_red_fused__native_batch_norm_legit_functional_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 21952
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 448
    x1 = (xindex // 448)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (448*r2) + (57344*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ty/ctyqwpqyjtqjgtoxvmiqmwf3lrkbf7b7jyfxjiaqma6oaszboymt.py
# Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional]
# out_57 => add_368, add_369, add_370, mul_463, mul_464, mul_465, mul_466, mul_467, rsqrt_66, squeeze_199, var_mean_66
triton_per_fused__native_batch_norm_legit_functional_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_64', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (448*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (448*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (448*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/74/c74pu73phrnopuqkaif3jlosi7xbjcmdnd3msmn5bmpdrp74m466.py
# Source Nodes: [out_57, out_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_57 => add_368, add_371, mul_462, mul_468, rsqrt_66, sub_66, var_mean_66
# out_58 => relu_64
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 448
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


# kernel path: /tmp/torchinductor_youkaichao/bb/cbbd7aniuuysaqerfskqsylpmxjk66xq6sqrsw6fndw3oln2aeei.py
# Source Nodes: [sp_204], Original ATen: [aten.convolution]
# sp_204 => convolution_67
triton_poi_fused_convolution_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (56*x2) + (10976*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ty/ctyj5acscpai57csgv6jddlfzjkdaebusxh5cnhzkyjvtrlbkl7p.py
# Source Nodes: [sp_205], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_205 => var_mean_67
triton_red_fused__native_batch_norm_legit_functional_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 728
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 56)
    x0 = xindex % 56
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
        tmp3 = tl.load(in_ptr0 + (x0 + (56*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/it/citkmfgsg3i6ttewckvfw6uzbup3eim42zuvtntliz5ctthfac2d.py
# Source Nodes: [sp_205], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_205 => add_373, add_374, add_375, mul_470, mul_471, mul_472, mul_473, mul_474, rsqrt_67, squeeze_202, var_mean_67
triton_per_fused__native_batch_norm_legit_functional_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_68', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 56
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (56*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (56*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (56*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/w6/cw66nsgjfg3du2jju7db4j2a2mpkaah7psp7didvyvudcwnkasd3.py
# Source Nodes: [sp_205, sp_206], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# sp_205 => add_373, add_376, mul_469, mul_475, rsqrt_67, sub_67, var_mean_67
# sp_206 => relu_65
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (y0 + (56*x2) + (10976*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (196*y0) + (87808*y1)), tmp14, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (56*x2) + (10976*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yb/cybhc6mmnvblzeh6cjlhl4pkkf4ixtvlhhstvlgokomdxypusxxf.py
# Source Nodes: [getattr_l__mod___layer3___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer3___0___pool => avg_pool2d_2
triton_poi_fused_avg_pool2d_70 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 14)
    x2 = xindex % 14
    y0 = yindex % 56
    y1 = (yindex // 56)
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
    tmp11 = tl.load(in_ptr0 + ((-12600) + y0 + (896*x2) + (25088*x3) + (351232*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x2
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-12152) + y0 + (896*x2) + (25088*x3) + (351232*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x2)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-11704) + y0 + (896*x2) + (25088*x3) + (351232*y1)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x3
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-56) + y0 + (896*x2) + (25088*x3) + (351232*y1)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (392 + y0 + (896*x2) + (25088*x3) + (351232*y1)), tmp41 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (840 + y0 + (896*x2) + (25088*x3) + (351232*y1)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x3)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (12488 + y0 + (896*x2) + (25088*x3) + (351232*y1)), tmp55 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (12936 + y0 + (896*x2) + (25088*x3) + (351232*y1)), tmp60 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (13384 + y0 + (896*x2) + (25088*x3) + (351232*y1)), tmp65 & xmask & ymask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x5 + (196*y0) + (87808*y1)), tmp178, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nz/cnzm6326kgnmixqy2cuyouqxuxu66lb6rdmzsbz5mwxvxlilmw4q.py
# Source Nodes: [cat_24], Original ATen: [aten.cat]
# cat_24 => cat_7
triton_poi_fused_cat_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3584
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 448
    y1 = (yindex // 448)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (448*x2) + (87808*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j6/cj6dc6dpvnjmy7toqk2qczotljzvabhvskghrfovwzdi5o6lzjud.py
# Source Nodes: [out_60], Original ATen: [aten.convolution]
# out_60 => convolution_74
triton_poi_fused_convolution_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_72', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/7r/c7rr7n2qrajv44rhm5y4gwwkfvzjy7a5oigc33y75efo7c5hddun.py
# Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
# out_61 => var_mean_74
triton_red_fused__native_batch_norm_legit_functional_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_73', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/at/catzztiax5lx7cf3f6czyofqanbjde7cp6ub42xh3d5gc6e2x3en.py
# Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
# out_61 => add_408, add_409, add_410, mul_519, mul_520, mul_521, mul_522, mul_523, rsqrt_74, squeeze_223, var_mean_74
triton_per_fused__native_batch_norm_legit_functional_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_74', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/mc/cmc5mwq4byug2idlkwrnmhxewu2vlniubd763bcspeecwsis6g3b.py
# Source Nodes: [out_61, out_62, shortcut_10, shortcut_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_61 => add_408, add_411, mul_518, mul_524, rsqrt_74, sub_74, var_mean_74
# out_62 => add_417
# shortcut_10 => add_413, add_416, mul_525, mul_531, rsqrt_75, sub_75, var_mean_75
# shortcut_11 => relu_72
triton_poi_fused__native_batch_norm_legit_functional_add_relu_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_75', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/lr/clrgqfs6h3jj6tzfiizu4fwtmiyieaeqmateksgr7ioueoceh6px.py
# Source Nodes: [out_65], Original ATen: [aten._native_batch_norm_legit_functional]
# out_65 => var_mean_76
triton_red_fused__native_batch_norm_legit_functional_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5824
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 448)
    x0 = xindex % 448
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
        tmp3 = tl.load(in_ptr0 + (x0 + (448*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ry/cryud3bswwnytovm5o5qligqz6vljjgafj5bzdqmnnayyqgmhk3d.py
# Source Nodes: [out_65], Original ATen: [aten._native_batch_norm_legit_functional]
# out_65 => add_419, add_420, add_421, mul_533, mul_534, mul_535, mul_536, mul_537, rsqrt_76, squeeze_229, var_mean_76
triton_per_fused__native_batch_norm_legit_functional_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_77', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (448*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (448*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (448*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/3f/c3fcgexsptledqv5vieaazkmqmzd4eeynkky655u5su4ofbdrw7y.py
# Source Nodes: [out_65, out_66], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_65 => add_419, add_422, mul_532, mul_538, rsqrt_76, sub_76, var_mean_76
# out_66 => relu_73
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 448
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


# kernel path: /tmp/torchinductor_youkaichao/mw/cmw5kqt3w5rn6wxom7prnjh5w46zd46eokeeh4pezpynoz5tiooj.py
# Source Nodes: [sp_236], Original ATen: [aten.add]
# sp_236 => add_428
triton_poi_fused_add_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 56
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (56 + x2 + (448*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (56*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xu/cxuipgphh6vjrcywfxzpbghzyeuw7yityhu2dxnm22vnhfwkvpxq.py
# Source Nodes: [sp_240], Original ATen: [aten.add]
# sp_240 => add_434
triton_poi_fused_add_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 56
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (112 + x2 + (448*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (56*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ey/ceygoqzleosexiiaw5cjuyyyl3qdqoa5atczqnoaa6xuvnxt3dd6.py
# Source Nodes: [sp_244], Original ATen: [aten.add]
# sp_244 => add_440
triton_poi_fused_add_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_81', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 56
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (168 + x2 + (448*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (56*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3n/c3nl24g4jan5kg7jz2hfx5f42lt2bxrstwkpz27pmny7hjt5m4ck.py
# Source Nodes: [sp_248], Original ATen: [aten.add]
# sp_248 => add_446
triton_poi_fused_add_82 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 56
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (224 + x2 + (448*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (56*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uq/cuqaigksm5gulkisbk6wodraml5n2yizkdcmrmsncgruisbox7bq.py
# Source Nodes: [sp_252], Original ATen: [aten.add]
# sp_252 => add_452
triton_poi_fused_add_83 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_83', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 56
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (280 + x2 + (448*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (56*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kl/cklgud2lwma7eavceivhdyuq5y6ewlnhxkot7eckxuuzqz2utv4v.py
# Source Nodes: [sp_256], Original ATen: [aten.add]
# sp_256 => add_458
triton_poi_fused_add_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 56
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (336 + x2 + (448*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (56*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ou/cou4mvyo366isvrkwxccfuq5k7c5wqzdjjftys4dgxexlwk6tsvu.py
# Source Nodes: [cat_23], Original ATen: [aten.cat]
# cat_23 => cat_8
triton_poi_fused_cat_85 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_85', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (392 + y0 + (448*x2) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (196*y0) + (87808*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pu/cpuzuirhskilojea54liyjz4wnt5x5lqyoanizs6iw4jvrm5pm2r.py
# Source Nodes: [out_69, out_70, shortcut_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_69 => add_465, add_468, mul_588, mul_594, rsqrt_84, sub_84, var_mean_84
# out_70 => add_469
# shortcut_12 => relu_81
triton_poi_fused__native_batch_norm_legit_functional_add_relu_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_86', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/wy/cwyjomo7o7ci3omajrtdgwciivuhacj5h525jhq7ogxcpuvkdkav.py
# Source Nodes: [out_104], Original ATen: [aten.convolution]
# out_104 => convolution_121
triton_poi_fused_convolution_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_87', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7168
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 896
    y1 = (yindex // 896)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (896*x2) + (175616*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ig/ciggpi4kvxbflg2onfhn3h2ikzvg3ci7xbqopr72mwbz2biljkhu.py
# Source Nodes: [out_105], Original ATen: [aten._native_batch_norm_legit_functional]
# out_105 => var_mean_121
triton_red_fused__native_batch_norm_legit_functional_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_88', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 11648
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 896)
    x0 = xindex % 896
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
        tmp3 = tl.load(in_ptr0 + (x0 + (896*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qf/cqffd4qyjnshj6d4r3jxwg43ipgfihl4dw5pc5baa6ri7bqxp3i7.py
# Source Nodes: [out_105], Original ATen: [aten._native_batch_norm_legit_functional]
# out_105 => add_679, add_680, add_681, mul_848, mul_849, mul_850, mul_851, mul_852, rsqrt_121, squeeze_364, var_mean_121
triton_per_fused__native_batch_norm_legit_functional_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_89', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (896*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (896*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (896*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4u/c4urwow63fkahhs3dwjb4xzxowwurgkvvtl5jebzos4bg73gmout.py
# Source Nodes: [out_105, out_106], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_105 => add_679, add_682, mul_847, mul_853, rsqrt_121, sub_121, var_mean_121
# out_106 => relu_118
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_90', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 896
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


# kernel path: /tmp/torchinductor_youkaichao/pl/cplx6zr37ybjzvwrilyukacmm7syyhn57kh6chnsg5c6okdzyxwh.py
# Source Nodes: [sp_378], Original ATen: [aten.convolution]
# sp_378 => convolution_122
triton_poi_fused_convolution_91 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_91', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (5488*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zj/czj7utv2feia46mu6ndnomezhary6qj3pmw4dc4ng6t6qwmzhweq.py
# Source Nodes: [sp_379], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_379 => var_mean_122
triton_red_fused__native_batch_norm_legit_functional_92 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_92', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 112
    x1 = (xindex // 112)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (112*r2) + (10976*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/6x/c6xrryqx3m4czvqih6rvscphufb7lmxuvdu565wyjvwfzjynns2a.py
# Source Nodes: [sp_379], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_379 => add_684, add_685, add_686, mul_855, mul_856, mul_857, mul_858, mul_859, rsqrt_122, squeeze_367, var_mean_122
triton_per_fused__native_batch_norm_legit_functional_93 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_93', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (112*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (112*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (112*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/yy/cyya2v7s6bs5woqlyggxmohbeihfz3ov7ey36b7tycqbe6rn6bqx.py
# Source Nodes: [sp_379, sp_380], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# sp_379 => add_684, add_687, mul_854, mul_860, rsqrt_122, sub_122, var_mean_122
# sp_380 => relu_119
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (y0 + (112*x2) + (5488*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (49*y0) + (43904*y1)), tmp14, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (112*x2) + (5488*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yf/cyftc4oygmwcea7zlyx6yqnzyyxtpa6jh7apxt2g6tzwa7k5t7k2.py
# Source Nodes: [getattr_l__mod___layer4___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer4___0___pool => avg_pool2d_3
triton_poi_fused_avg_pool2d_95 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_95', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 7)
    x2 = xindex % 7
    y0 = yindex % 112
    y1 = (yindex // 112)
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
    tmp11 = tl.load(in_ptr0 + ((-12656) + y0 + (1792*x2) + (25088*x3) + (175616*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x2
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-11760) + y0 + (1792*x2) + (25088*x3) + (175616*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x2)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-10864) + y0 + (1792*x2) + (25088*x3) + (175616*y1)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x3
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-112) + y0 + (1792*x2) + (25088*x3) + (175616*y1)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (784 + y0 + (1792*x2) + (25088*x3) + (175616*y1)), tmp41 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1680 + y0 + (1792*x2) + (25088*x3) + (175616*y1)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x3)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (12432 + y0 + (1792*x2) + (25088*x3) + (175616*y1)), tmp55 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (13328 + y0 + (1792*x2) + (25088*x3) + (175616*y1)), tmp60 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (14224 + y0 + (1792*x2) + (25088*x3) + (175616*y1)), tmp65 & xmask & ymask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x5 + (49*y0) + (43904*y1)), tmp178, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u3/cu3u4ynwkb2pj5bhwpdjx7q4pj3vazwkwgzjtjflddbyve2cvntw.py
# Source Nodes: [cat_18], Original ATen: [aten.cat]
# cat_18 => cat_13
triton_poi_fused_cat_96 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_96', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7168
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 896
    y1 = (yindex // 896)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (896*x2) + (43904*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wd/cwdfvteu6buynv36vipceu6n3vpqd2xdnyvknqa7bg6u7awhwsl5.py
# Source Nodes: [out_108], Original ATen: [aten.convolution]
# out_108 => convolution_129
triton_poi_fused_convolution_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_97', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/6z/c6zffvpmqdq32rsree4tcnoqjdhjboz63iycmzeepqwoyktm4jig.py
# Source Nodes: [out_109], Original ATen: [aten._native_batch_norm_legit_functional]
# out_109 => var_mean_129
triton_red_fused__native_batch_norm_legit_functional_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_98', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/oa/coarpct6kasnpnevqwwuhyoso6nevcsgxl2ne3octumncnxpqhco.py
# Source Nodes: [out_109], Original ATen: [aten._native_batch_norm_legit_functional]
# out_109 => add_719, add_720, add_721, mul_904, mul_905, mul_906, mul_907, mul_908, rsqrt_129, squeeze_388, var_mean_129
triton_per_fused__native_batch_norm_legit_functional_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_99', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/lb/clbnjmnzc6u7oj3o6b3lpn3bqbvo2dd757qkiyiezl3gq7unovbu.py
# Source Nodes: [out_109, out_110, shortcut_17, shortcut_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_109 => add_719, add_722, mul_903, mul_909, rsqrt_129, sub_129, var_mean_129
# out_110 => add_728
# shortcut_17 => add_724, add_727, mul_910, mul_916, rsqrt_130, sub_130, var_mean_130
# shortcut_18 => relu_126
triton_poi_fused__native_batch_norm_legit_functional_add_relu_100 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_100', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/zt/cztdyndlpwxbj4mar5cgizyeszek4rgnxii3co2zkohg52rmzthc.py
# Source Nodes: [out_113], Original ATen: [aten._native_batch_norm_legit_functional]
# out_113 => var_mean_131
triton_red_fused__native_batch_norm_legit_functional_101 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_101', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3584
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 896
    x1 = (xindex // 896)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (896*r2) + (87808*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/36/c36otbbdodcpvf3vp5oaleh7diqzlz2hwisuecyj2nf27g6qw4ps.py
# Source Nodes: [out_113], Original ATen: [aten._native_batch_norm_legit_functional]
# out_113 => add_730, add_731, add_732, mul_918, mul_919, mul_920, mul_921, mul_922, rsqrt_131, squeeze_394, var_mean_131
triton_per_fused__native_batch_norm_legit_functional_102 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_102', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (896*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (896*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (896*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4c/c4chuzezpzvmphd26iesjvph3mzmcc6zfshyuvr3dkkq77d2syu6.py
# Source Nodes: [out_113, out_114], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_113 => add_730, add_733, mul_917, mul_923, rsqrt_131, sub_131, var_mean_131
# out_114 => relu_127
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_103 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_103', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 896
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


# kernel path: /tmp/torchinductor_youkaichao/wl/cwlp6uhg5r5dxaa6cd5abjec2rknlo4yhevge7s2xb3dqsxus5fq.py
# Source Nodes: [sp_410], Original ATen: [aten.add]
# sp_410 => add_739
triton_poi_fused_add_104 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_104', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 112
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (112 + x2 + (896*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (112*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o2/co2njy34ikksl6cqiixy4thluabyxr2pxoq5vmzb6hs6esctrrsb.py
# Source Nodes: [sp_414], Original ATen: [aten.add]
# sp_414 => add_745
triton_poi_fused_add_105 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_105', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 112
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (224 + x2 + (896*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (112*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gy/cgyvxlfm2rsz3l5vqiqpdxywbwa5abxsh5adqz34b3v3iwxxooa3.py
# Source Nodes: [sp_418], Original ATen: [aten.add]
# sp_418 => add_751
triton_poi_fused_add_106 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_106', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 112
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (336 + x2 + (896*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (112*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oc/coczq3vtfwu2k5xsde6gmzk7y2zyqr5armcnuyp62a47s77spvwk.py
# Source Nodes: [sp_422], Original ATen: [aten.add]
# sp_422 => add_757
triton_poi_fused_add_107 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_107', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 112
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (448 + x2 + (896*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (112*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yq/cyq6oqodxbqe2bzspzm6tdqif6nrlotan2dakx7nbymwwckqbowf.py
# Source Nodes: [sp_426], Original ATen: [aten.add]
# sp_426 => add_763
triton_poi_fused_add_108 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_108', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 112
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (560 + x2 + (896*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (112*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uz/cuz7z35laxrjrhpfps5fcbarcb2ugsuqhva4yusaimf7pz66bnqk.py
# Source Nodes: [sp_430], Original ATen: [aten.add]
# sp_430 => add_769
triton_poi_fused_add_109 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_109', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 112
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (672 + x2 + (896*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (112*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cd/ccdqt7pmu36oqhbrjfqfnh7rdb2gbcceh7xbdn567zokzoyoz2tt.py
# Source Nodes: [cat_17], Original ATen: [aten.cat]
# cat_17 => cat_14
triton_poi_fused_cat_110 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_110', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (784 + y0 + (896*x2) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (49*y0) + (43904*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xh/cxhgnvynoixykcddpt2sysuwtrvm3ymfl3pvezqb52xovxawyax7.py
# Source Nodes: [out_117, out_118, shortcut_19], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_117 => add_776, add_779, mul_973, mul_979, rsqrt_139, sub_139, var_mean_139
# out_118 => add_780
# shortcut_19 => relu_135
triton_poi_fused__native_batch_norm_legit_functional_add_relu_111 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_111', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/u6/cu6crg24eb4bayy63ia6vynmmqnq76z7g2tc4y3b6ammishy2rop.py
# Source Nodes: [out_125, out_126, x_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
# out_125 => add_828, add_831, mul_1036, mul_1042, rsqrt_148, sub_148, var_mean_148
# out_126 => add_832
# x_8 => relu_144
triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_112 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_112', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/gt/cgtplzpbogbeycsnwke6cime6gb25hpvu7h57nlmtkvf5yuitz4i.py
# Source Nodes: [x_11, x_9], Original ATen: [aten.mean, aten.view]
# x_11 => view
# x_9 => mean
triton_per_fused_mean_view_113 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_view_113', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/4e/c4etg7cqum3l6tnt37whr2hb6r725s3s4ge4lapetw5ga35zz23z.py
# Source Nodes: [x_1], Original ATen: [aten.add]
# x_1 => add
triton_poi_fused_add_114 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_114', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (112, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_5, (112, ), (1, ))
    assert_size_stride(primals_6, (112, ), (1, ))
    assert_size_stride(primals_7, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_8, (14, ), (1, ))
    assert_size_stride(primals_9, (14, ), (1, ))
    assert_size_stride(primals_10, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_11, (14, ), (1, ))
    assert_size_stride(primals_12, (14, ), (1, ))
    assert_size_stride(primals_13, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_14, (14, ), (1, ))
    assert_size_stride(primals_15, (14, ), (1, ))
    assert_size_stride(primals_16, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_17, (14, ), (1, ))
    assert_size_stride(primals_18, (14, ), (1, ))
    assert_size_stride(primals_19, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_20, (14, ), (1, ))
    assert_size_stride(primals_21, (14, ), (1, ))
    assert_size_stride(primals_22, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_23, (14, ), (1, ))
    assert_size_stride(primals_24, (14, ), (1, ))
    assert_size_stride(primals_25, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_26, (14, ), (1, ))
    assert_size_stride(primals_27, (14, ), (1, ))
    assert_size_stride(primals_28, (256, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_30, (256, ), (1, ))
    assert_size_stride(primals_31, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_34, (112, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_35, (112, ), (1, ))
    assert_size_stride(primals_36, (112, ), (1, ))
    assert_size_stride(primals_37, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_38, (14, ), (1, ))
    assert_size_stride(primals_39, (14, ), (1, ))
    assert_size_stride(primals_40, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_41, (14, ), (1, ))
    assert_size_stride(primals_42, (14, ), (1, ))
    assert_size_stride(primals_43, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_44, (14, ), (1, ))
    assert_size_stride(primals_45, (14, ), (1, ))
    assert_size_stride(primals_46, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_47, (14, ), (1, ))
    assert_size_stride(primals_48, (14, ), (1, ))
    assert_size_stride(primals_49, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_50, (14, ), (1, ))
    assert_size_stride(primals_51, (14, ), (1, ))
    assert_size_stride(primals_52, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_53, (14, ), (1, ))
    assert_size_stride(primals_54, (14, ), (1, ))
    assert_size_stride(primals_55, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_56, (14, ), (1, ))
    assert_size_stride(primals_57, (14, ), (1, ))
    assert_size_stride(primals_58, (256, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_59, (256, ), (1, ))
    assert_size_stride(primals_60, (256, ), (1, ))
    assert_size_stride(primals_61, (112, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_62, (112, ), (1, ))
    assert_size_stride(primals_63, (112, ), (1, ))
    assert_size_stride(primals_64, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_65, (14, ), (1, ))
    assert_size_stride(primals_66, (14, ), (1, ))
    assert_size_stride(primals_67, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_68, (14, ), (1, ))
    assert_size_stride(primals_69, (14, ), (1, ))
    assert_size_stride(primals_70, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_71, (14, ), (1, ))
    assert_size_stride(primals_72, (14, ), (1, ))
    assert_size_stride(primals_73, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_74, (14, ), (1, ))
    assert_size_stride(primals_75, (14, ), (1, ))
    assert_size_stride(primals_76, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_77, (14, ), (1, ))
    assert_size_stride(primals_78, (14, ), (1, ))
    assert_size_stride(primals_79, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_80, (14, ), (1, ))
    assert_size_stride(primals_81, (14, ), (1, ))
    assert_size_stride(primals_82, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(primals_83, (14, ), (1, ))
    assert_size_stride(primals_84, (14, ), (1, ))
    assert_size_stride(primals_85, (256, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_87, (256, ), (1, ))
    assert_size_stride(primals_88, (224, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_89, (224, ), (1, ))
    assert_size_stride(primals_90, (224, ), (1, ))
    assert_size_stride(primals_91, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_92, (28, ), (1, ))
    assert_size_stride(primals_93, (28, ), (1, ))
    assert_size_stride(primals_94, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_95, (28, ), (1, ))
    assert_size_stride(primals_96, (28, ), (1, ))
    assert_size_stride(primals_97, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_98, (28, ), (1, ))
    assert_size_stride(primals_99, (28, ), (1, ))
    assert_size_stride(primals_100, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_101, (28, ), (1, ))
    assert_size_stride(primals_102, (28, ), (1, ))
    assert_size_stride(primals_103, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_104, (28, ), (1, ))
    assert_size_stride(primals_105, (28, ), (1, ))
    assert_size_stride(primals_106, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_107, (28, ), (1, ))
    assert_size_stride(primals_108, (28, ), (1, ))
    assert_size_stride(primals_109, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_110, (28, ), (1, ))
    assert_size_stride(primals_111, (28, ), (1, ))
    assert_size_stride(primals_112, (512, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_113, (512, ), (1, ))
    assert_size_stride(primals_114, (512, ), (1, ))
    assert_size_stride(primals_115, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_116, (512, ), (1, ))
    assert_size_stride(primals_117, (512, ), (1, ))
    assert_size_stride(primals_118, (224, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_119, (224, ), (1, ))
    assert_size_stride(primals_120, (224, ), (1, ))
    assert_size_stride(primals_121, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_122, (28, ), (1, ))
    assert_size_stride(primals_123, (28, ), (1, ))
    assert_size_stride(primals_124, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_125, (28, ), (1, ))
    assert_size_stride(primals_126, (28, ), (1, ))
    assert_size_stride(primals_127, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_128, (28, ), (1, ))
    assert_size_stride(primals_129, (28, ), (1, ))
    assert_size_stride(primals_130, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_131, (28, ), (1, ))
    assert_size_stride(primals_132, (28, ), (1, ))
    assert_size_stride(primals_133, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_134, (28, ), (1, ))
    assert_size_stride(primals_135, (28, ), (1, ))
    assert_size_stride(primals_136, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_137, (28, ), (1, ))
    assert_size_stride(primals_138, (28, ), (1, ))
    assert_size_stride(primals_139, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_140, (28, ), (1, ))
    assert_size_stride(primals_141, (28, ), (1, ))
    assert_size_stride(primals_142, (512, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_144, (512, ), (1, ))
    assert_size_stride(primals_145, (224, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_146, (224, ), (1, ))
    assert_size_stride(primals_147, (224, ), (1, ))
    assert_size_stride(primals_148, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_149, (28, ), (1, ))
    assert_size_stride(primals_150, (28, ), (1, ))
    assert_size_stride(primals_151, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_152, (28, ), (1, ))
    assert_size_stride(primals_153, (28, ), (1, ))
    assert_size_stride(primals_154, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_155, (28, ), (1, ))
    assert_size_stride(primals_156, (28, ), (1, ))
    assert_size_stride(primals_157, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_158, (28, ), (1, ))
    assert_size_stride(primals_159, (28, ), (1, ))
    assert_size_stride(primals_160, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_161, (28, ), (1, ))
    assert_size_stride(primals_162, (28, ), (1, ))
    assert_size_stride(primals_163, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_164, (28, ), (1, ))
    assert_size_stride(primals_165, (28, ), (1, ))
    assert_size_stride(primals_166, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_167, (28, ), (1, ))
    assert_size_stride(primals_168, (28, ), (1, ))
    assert_size_stride(primals_169, (512, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_170, (512, ), (1, ))
    assert_size_stride(primals_171, (512, ), (1, ))
    assert_size_stride(primals_172, (224, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_173, (224, ), (1, ))
    assert_size_stride(primals_174, (224, ), (1, ))
    assert_size_stride(primals_175, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_176, (28, ), (1, ))
    assert_size_stride(primals_177, (28, ), (1, ))
    assert_size_stride(primals_178, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_179, (28, ), (1, ))
    assert_size_stride(primals_180, (28, ), (1, ))
    assert_size_stride(primals_181, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_182, (28, ), (1, ))
    assert_size_stride(primals_183, (28, ), (1, ))
    assert_size_stride(primals_184, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_185, (28, ), (1, ))
    assert_size_stride(primals_186, (28, ), (1, ))
    assert_size_stride(primals_187, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_188, (28, ), (1, ))
    assert_size_stride(primals_189, (28, ), (1, ))
    assert_size_stride(primals_190, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_191, (28, ), (1, ))
    assert_size_stride(primals_192, (28, ), (1, ))
    assert_size_stride(primals_193, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(primals_194, (28, ), (1, ))
    assert_size_stride(primals_195, (28, ), (1, ))
    assert_size_stride(primals_196, (512, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_197, (512, ), (1, ))
    assert_size_stride(primals_198, (512, ), (1, ))
    assert_size_stride(primals_199, (448, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_200, (448, ), (1, ))
    assert_size_stride(primals_201, (448, ), (1, ))
    assert_size_stride(primals_202, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_203, (56, ), (1, ))
    assert_size_stride(primals_204, (56, ), (1, ))
    assert_size_stride(primals_205, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_206, (56, ), (1, ))
    assert_size_stride(primals_207, (56, ), (1, ))
    assert_size_stride(primals_208, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_209, (56, ), (1, ))
    assert_size_stride(primals_210, (56, ), (1, ))
    assert_size_stride(primals_211, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_212, (56, ), (1, ))
    assert_size_stride(primals_213, (56, ), (1, ))
    assert_size_stride(primals_214, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_215, (56, ), (1, ))
    assert_size_stride(primals_216, (56, ), (1, ))
    assert_size_stride(primals_217, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_218, (56, ), (1, ))
    assert_size_stride(primals_219, (56, ), (1, ))
    assert_size_stride(primals_220, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_221, (56, ), (1, ))
    assert_size_stride(primals_222, (56, ), (1, ))
    assert_size_stride(primals_223, (1024, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_224, (1024, ), (1, ))
    assert_size_stride(primals_225, (1024, ), (1, ))
    assert_size_stride(primals_226, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_227, (1024, ), (1, ))
    assert_size_stride(primals_228, (1024, ), (1, ))
    assert_size_stride(primals_229, (448, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_230, (448, ), (1, ))
    assert_size_stride(primals_231, (448, ), (1, ))
    assert_size_stride(primals_232, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_233, (56, ), (1, ))
    assert_size_stride(primals_234, (56, ), (1, ))
    assert_size_stride(primals_235, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_236, (56, ), (1, ))
    assert_size_stride(primals_237, (56, ), (1, ))
    assert_size_stride(primals_238, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_239, (56, ), (1, ))
    assert_size_stride(primals_240, (56, ), (1, ))
    assert_size_stride(primals_241, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_242, (56, ), (1, ))
    assert_size_stride(primals_243, (56, ), (1, ))
    assert_size_stride(primals_244, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_245, (56, ), (1, ))
    assert_size_stride(primals_246, (56, ), (1, ))
    assert_size_stride(primals_247, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_248, (56, ), (1, ))
    assert_size_stride(primals_249, (56, ), (1, ))
    assert_size_stride(primals_250, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_251, (56, ), (1, ))
    assert_size_stride(primals_252, (56, ), (1, ))
    assert_size_stride(primals_253, (1024, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_254, (1024, ), (1, ))
    assert_size_stride(primals_255, (1024, ), (1, ))
    assert_size_stride(primals_256, (448, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_257, (448, ), (1, ))
    assert_size_stride(primals_258, (448, ), (1, ))
    assert_size_stride(primals_259, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_260, (56, ), (1, ))
    assert_size_stride(primals_261, (56, ), (1, ))
    assert_size_stride(primals_262, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_263, (56, ), (1, ))
    assert_size_stride(primals_264, (56, ), (1, ))
    assert_size_stride(primals_265, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_266, (56, ), (1, ))
    assert_size_stride(primals_267, (56, ), (1, ))
    assert_size_stride(primals_268, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_269, (56, ), (1, ))
    assert_size_stride(primals_270, (56, ), (1, ))
    assert_size_stride(primals_271, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_272, (56, ), (1, ))
    assert_size_stride(primals_273, (56, ), (1, ))
    assert_size_stride(primals_274, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_275, (56, ), (1, ))
    assert_size_stride(primals_276, (56, ), (1, ))
    assert_size_stride(primals_277, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_278, (56, ), (1, ))
    assert_size_stride(primals_279, (56, ), (1, ))
    assert_size_stride(primals_280, (1024, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_281, (1024, ), (1, ))
    assert_size_stride(primals_282, (1024, ), (1, ))
    assert_size_stride(primals_283, (448, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_284, (448, ), (1, ))
    assert_size_stride(primals_285, (448, ), (1, ))
    assert_size_stride(primals_286, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_287, (56, ), (1, ))
    assert_size_stride(primals_288, (56, ), (1, ))
    assert_size_stride(primals_289, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_290, (56, ), (1, ))
    assert_size_stride(primals_291, (56, ), (1, ))
    assert_size_stride(primals_292, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_293, (56, ), (1, ))
    assert_size_stride(primals_294, (56, ), (1, ))
    assert_size_stride(primals_295, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_296, (56, ), (1, ))
    assert_size_stride(primals_297, (56, ), (1, ))
    assert_size_stride(primals_298, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_299, (56, ), (1, ))
    assert_size_stride(primals_300, (56, ), (1, ))
    assert_size_stride(primals_301, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_302, (56, ), (1, ))
    assert_size_stride(primals_303, (56, ), (1, ))
    assert_size_stride(primals_304, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_305, (56, ), (1, ))
    assert_size_stride(primals_306, (56, ), (1, ))
    assert_size_stride(primals_307, (1024, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_308, (1024, ), (1, ))
    assert_size_stride(primals_309, (1024, ), (1, ))
    assert_size_stride(primals_310, (448, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_311, (448, ), (1, ))
    assert_size_stride(primals_312, (448, ), (1, ))
    assert_size_stride(primals_313, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_314, (56, ), (1, ))
    assert_size_stride(primals_315, (56, ), (1, ))
    assert_size_stride(primals_316, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_317, (56, ), (1, ))
    assert_size_stride(primals_318, (56, ), (1, ))
    assert_size_stride(primals_319, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_320, (56, ), (1, ))
    assert_size_stride(primals_321, (56, ), (1, ))
    assert_size_stride(primals_322, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_323, (56, ), (1, ))
    assert_size_stride(primals_324, (56, ), (1, ))
    assert_size_stride(primals_325, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_326, (56, ), (1, ))
    assert_size_stride(primals_327, (56, ), (1, ))
    assert_size_stride(primals_328, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_329, (56, ), (1, ))
    assert_size_stride(primals_330, (56, ), (1, ))
    assert_size_stride(primals_331, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_332, (56, ), (1, ))
    assert_size_stride(primals_333, (56, ), (1, ))
    assert_size_stride(primals_334, (1024, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_335, (1024, ), (1, ))
    assert_size_stride(primals_336, (1024, ), (1, ))
    assert_size_stride(primals_337, (448, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_338, (448, ), (1, ))
    assert_size_stride(primals_339, (448, ), (1, ))
    assert_size_stride(primals_340, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_341, (56, ), (1, ))
    assert_size_stride(primals_342, (56, ), (1, ))
    assert_size_stride(primals_343, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_344, (56, ), (1, ))
    assert_size_stride(primals_345, (56, ), (1, ))
    assert_size_stride(primals_346, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_347, (56, ), (1, ))
    assert_size_stride(primals_348, (56, ), (1, ))
    assert_size_stride(primals_349, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_350, (56, ), (1, ))
    assert_size_stride(primals_351, (56, ), (1, ))
    assert_size_stride(primals_352, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_353, (56, ), (1, ))
    assert_size_stride(primals_354, (56, ), (1, ))
    assert_size_stride(primals_355, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_356, (56, ), (1, ))
    assert_size_stride(primals_357, (56, ), (1, ))
    assert_size_stride(primals_358, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_359, (56, ), (1, ))
    assert_size_stride(primals_360, (56, ), (1, ))
    assert_size_stride(primals_361, (1024, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_362, (1024, ), (1, ))
    assert_size_stride(primals_363, (1024, ), (1, ))
    assert_size_stride(primals_364, (896, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_365, (896, ), (1, ))
    assert_size_stride(primals_366, (896, ), (1, ))
    assert_size_stride(primals_367, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_368, (112, ), (1, ))
    assert_size_stride(primals_369, (112, ), (1, ))
    assert_size_stride(primals_370, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_371, (112, ), (1, ))
    assert_size_stride(primals_372, (112, ), (1, ))
    assert_size_stride(primals_373, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_374, (112, ), (1, ))
    assert_size_stride(primals_375, (112, ), (1, ))
    assert_size_stride(primals_376, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_377, (112, ), (1, ))
    assert_size_stride(primals_378, (112, ), (1, ))
    assert_size_stride(primals_379, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_380, (112, ), (1, ))
    assert_size_stride(primals_381, (112, ), (1, ))
    assert_size_stride(primals_382, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_383, (112, ), (1, ))
    assert_size_stride(primals_384, (112, ), (1, ))
    assert_size_stride(primals_385, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_386, (112, ), (1, ))
    assert_size_stride(primals_387, (112, ), (1, ))
    assert_size_stride(primals_388, (2048, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_389, (2048, ), (1, ))
    assert_size_stride(primals_390, (2048, ), (1, ))
    assert_size_stride(primals_391, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_392, (2048, ), (1, ))
    assert_size_stride(primals_393, (2048, ), (1, ))
    assert_size_stride(primals_394, (896, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_395, (896, ), (1, ))
    assert_size_stride(primals_396, (896, ), (1, ))
    assert_size_stride(primals_397, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_398, (112, ), (1, ))
    assert_size_stride(primals_399, (112, ), (1, ))
    assert_size_stride(primals_400, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_401, (112, ), (1, ))
    assert_size_stride(primals_402, (112, ), (1, ))
    assert_size_stride(primals_403, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_404, (112, ), (1, ))
    assert_size_stride(primals_405, (112, ), (1, ))
    assert_size_stride(primals_406, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_407, (112, ), (1, ))
    assert_size_stride(primals_408, (112, ), (1, ))
    assert_size_stride(primals_409, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_410, (112, ), (1, ))
    assert_size_stride(primals_411, (112, ), (1, ))
    assert_size_stride(primals_412, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_413, (112, ), (1, ))
    assert_size_stride(primals_414, (112, ), (1, ))
    assert_size_stride(primals_415, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_416, (112, ), (1, ))
    assert_size_stride(primals_417, (112, ), (1, ))
    assert_size_stride(primals_418, (2048, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_419, (2048, ), (1, ))
    assert_size_stride(primals_420, (2048, ), (1, ))
    assert_size_stride(primals_421, (896, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_422, (896, ), (1, ))
    assert_size_stride(primals_423, (896, ), (1, ))
    assert_size_stride(primals_424, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_425, (112, ), (1, ))
    assert_size_stride(primals_426, (112, ), (1, ))
    assert_size_stride(primals_427, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_428, (112, ), (1, ))
    assert_size_stride(primals_429, (112, ), (1, ))
    assert_size_stride(primals_430, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_431, (112, ), (1, ))
    assert_size_stride(primals_432, (112, ), (1, ))
    assert_size_stride(primals_433, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_434, (112, ), (1, ))
    assert_size_stride(primals_435, (112, ), (1, ))
    assert_size_stride(primals_436, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_437, (112, ), (1, ))
    assert_size_stride(primals_438, (112, ), (1, ))
    assert_size_stride(primals_439, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_440, (112, ), (1, ))
    assert_size_stride(primals_441, (112, ), (1, ))
    assert_size_stride(primals_442, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_443, (112, ), (1, ))
    assert_size_stride(primals_444, (112, ), (1, ))
    assert_size_stride(primals_445, (2048, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_446, (2048, ), (1, ))
    assert_size_stride(primals_447, (2048, ), (1, ))
    assert_size_stride(primals_448, (1000, 2048), (2048, 1))
    assert_size_stride(primals_449, (1000, ), (1, ))
    assert_size_stride(primals_450, (64, ), (1, ))
    assert_size_stride(primals_451, (64, ), (1, ))
    assert_size_stride(primals_452, (), ())
    assert_size_stride(primals_453, (112, ), (1, ))
    assert_size_stride(primals_454, (112, ), (1, ))
    assert_size_stride(primals_455, (), ())
    assert_size_stride(primals_456, (14, ), (1, ))
    assert_size_stride(primals_457, (14, ), (1, ))
    assert_size_stride(primals_458, (), ())
    assert_size_stride(primals_459, (14, ), (1, ))
    assert_size_stride(primals_460, (14, ), (1, ))
    assert_size_stride(primals_461, (), ())
    assert_size_stride(primals_462, (14, ), (1, ))
    assert_size_stride(primals_463, (14, ), (1, ))
    assert_size_stride(primals_464, (), ())
    assert_size_stride(primals_465, (14, ), (1, ))
    assert_size_stride(primals_466, (14, ), (1, ))
    assert_size_stride(primals_467, (), ())
    assert_size_stride(primals_468, (14, ), (1, ))
    assert_size_stride(primals_469, (14, ), (1, ))
    assert_size_stride(primals_470, (), ())
    assert_size_stride(primals_471, (14, ), (1, ))
    assert_size_stride(primals_472, (14, ), (1, ))
    assert_size_stride(primals_473, (), ())
    assert_size_stride(primals_474, (14, ), (1, ))
    assert_size_stride(primals_475, (14, ), (1, ))
    assert_size_stride(primals_476, (), ())
    assert_size_stride(primals_477, (256, ), (1, ))
    assert_size_stride(primals_478, (256, ), (1, ))
    assert_size_stride(primals_479, (), ())
    assert_size_stride(primals_480, (256, ), (1, ))
    assert_size_stride(primals_481, (256, ), (1, ))
    assert_size_stride(primals_482, (), ())
    assert_size_stride(primals_483, (112, ), (1, ))
    assert_size_stride(primals_484, (112, ), (1, ))
    assert_size_stride(primals_485, (), ())
    assert_size_stride(primals_486, (14, ), (1, ))
    assert_size_stride(primals_487, (14, ), (1, ))
    assert_size_stride(primals_488, (), ())
    assert_size_stride(primals_489, (14, ), (1, ))
    assert_size_stride(primals_490, (14, ), (1, ))
    assert_size_stride(primals_491, (), ())
    assert_size_stride(primals_492, (14, ), (1, ))
    assert_size_stride(primals_493, (14, ), (1, ))
    assert_size_stride(primals_494, (), ())
    assert_size_stride(primals_495, (14, ), (1, ))
    assert_size_stride(primals_496, (14, ), (1, ))
    assert_size_stride(primals_497, (), ())
    assert_size_stride(primals_498, (14, ), (1, ))
    assert_size_stride(primals_499, (14, ), (1, ))
    assert_size_stride(primals_500, (), ())
    assert_size_stride(primals_501, (14, ), (1, ))
    assert_size_stride(primals_502, (14, ), (1, ))
    assert_size_stride(primals_503, (), ())
    assert_size_stride(primals_504, (14, ), (1, ))
    assert_size_stride(primals_505, (14, ), (1, ))
    assert_size_stride(primals_506, (), ())
    assert_size_stride(primals_507, (256, ), (1, ))
    assert_size_stride(primals_508, (256, ), (1, ))
    assert_size_stride(primals_509, (), ())
    assert_size_stride(primals_510, (112, ), (1, ))
    assert_size_stride(primals_511, (112, ), (1, ))
    assert_size_stride(primals_512, (), ())
    assert_size_stride(primals_513, (14, ), (1, ))
    assert_size_stride(primals_514, (14, ), (1, ))
    assert_size_stride(primals_515, (), ())
    assert_size_stride(primals_516, (14, ), (1, ))
    assert_size_stride(primals_517, (14, ), (1, ))
    assert_size_stride(primals_518, (), ())
    assert_size_stride(primals_519, (14, ), (1, ))
    assert_size_stride(primals_520, (14, ), (1, ))
    assert_size_stride(primals_521, (), ())
    assert_size_stride(primals_522, (14, ), (1, ))
    assert_size_stride(primals_523, (14, ), (1, ))
    assert_size_stride(primals_524, (), ())
    assert_size_stride(primals_525, (14, ), (1, ))
    assert_size_stride(primals_526, (14, ), (1, ))
    assert_size_stride(primals_527, (), ())
    assert_size_stride(primals_528, (14, ), (1, ))
    assert_size_stride(primals_529, (14, ), (1, ))
    assert_size_stride(primals_530, (), ())
    assert_size_stride(primals_531, (14, ), (1, ))
    assert_size_stride(primals_532, (14, ), (1, ))
    assert_size_stride(primals_533, (), ())
    assert_size_stride(primals_534, (256, ), (1, ))
    assert_size_stride(primals_535, (256, ), (1, ))
    assert_size_stride(primals_536, (), ())
    assert_size_stride(primals_537, (224, ), (1, ))
    assert_size_stride(primals_538, (224, ), (1, ))
    assert_size_stride(primals_539, (), ())
    assert_size_stride(primals_540, (28, ), (1, ))
    assert_size_stride(primals_541, (28, ), (1, ))
    assert_size_stride(primals_542, (), ())
    assert_size_stride(primals_543, (28, ), (1, ))
    assert_size_stride(primals_544, (28, ), (1, ))
    assert_size_stride(primals_545, (), ())
    assert_size_stride(primals_546, (28, ), (1, ))
    assert_size_stride(primals_547, (28, ), (1, ))
    assert_size_stride(primals_548, (), ())
    assert_size_stride(primals_549, (28, ), (1, ))
    assert_size_stride(primals_550, (28, ), (1, ))
    assert_size_stride(primals_551, (), ())
    assert_size_stride(primals_552, (28, ), (1, ))
    assert_size_stride(primals_553, (28, ), (1, ))
    assert_size_stride(primals_554, (), ())
    assert_size_stride(primals_555, (28, ), (1, ))
    assert_size_stride(primals_556, (28, ), (1, ))
    assert_size_stride(primals_557, (), ())
    assert_size_stride(primals_558, (28, ), (1, ))
    assert_size_stride(primals_559, (28, ), (1, ))
    assert_size_stride(primals_560, (), ())
    assert_size_stride(primals_561, (512, ), (1, ))
    assert_size_stride(primals_562, (512, ), (1, ))
    assert_size_stride(primals_563, (), ())
    assert_size_stride(primals_564, (512, ), (1, ))
    assert_size_stride(primals_565, (512, ), (1, ))
    assert_size_stride(primals_566, (), ())
    assert_size_stride(primals_567, (224, ), (1, ))
    assert_size_stride(primals_568, (224, ), (1, ))
    assert_size_stride(primals_569, (), ())
    assert_size_stride(primals_570, (28, ), (1, ))
    assert_size_stride(primals_571, (28, ), (1, ))
    assert_size_stride(primals_572, (), ())
    assert_size_stride(primals_573, (28, ), (1, ))
    assert_size_stride(primals_574, (28, ), (1, ))
    assert_size_stride(primals_575, (), ())
    assert_size_stride(primals_576, (28, ), (1, ))
    assert_size_stride(primals_577, (28, ), (1, ))
    assert_size_stride(primals_578, (), ())
    assert_size_stride(primals_579, (28, ), (1, ))
    assert_size_stride(primals_580, (28, ), (1, ))
    assert_size_stride(primals_581, (), ())
    assert_size_stride(primals_582, (28, ), (1, ))
    assert_size_stride(primals_583, (28, ), (1, ))
    assert_size_stride(primals_584, (), ())
    assert_size_stride(primals_585, (28, ), (1, ))
    assert_size_stride(primals_586, (28, ), (1, ))
    assert_size_stride(primals_587, (), ())
    assert_size_stride(primals_588, (28, ), (1, ))
    assert_size_stride(primals_589, (28, ), (1, ))
    assert_size_stride(primals_590, (), ())
    assert_size_stride(primals_591, (512, ), (1, ))
    assert_size_stride(primals_592, (512, ), (1, ))
    assert_size_stride(primals_593, (), ())
    assert_size_stride(primals_594, (224, ), (1, ))
    assert_size_stride(primals_595, (224, ), (1, ))
    assert_size_stride(primals_596, (), ())
    assert_size_stride(primals_597, (28, ), (1, ))
    assert_size_stride(primals_598, (28, ), (1, ))
    assert_size_stride(primals_599, (), ())
    assert_size_stride(primals_600, (28, ), (1, ))
    assert_size_stride(primals_601, (28, ), (1, ))
    assert_size_stride(primals_602, (), ())
    assert_size_stride(primals_603, (28, ), (1, ))
    assert_size_stride(primals_604, (28, ), (1, ))
    assert_size_stride(primals_605, (), ())
    assert_size_stride(primals_606, (28, ), (1, ))
    assert_size_stride(primals_607, (28, ), (1, ))
    assert_size_stride(primals_608, (), ())
    assert_size_stride(primals_609, (28, ), (1, ))
    assert_size_stride(primals_610, (28, ), (1, ))
    assert_size_stride(primals_611, (), ())
    assert_size_stride(primals_612, (28, ), (1, ))
    assert_size_stride(primals_613, (28, ), (1, ))
    assert_size_stride(primals_614, (), ())
    assert_size_stride(primals_615, (28, ), (1, ))
    assert_size_stride(primals_616, (28, ), (1, ))
    assert_size_stride(primals_617, (), ())
    assert_size_stride(primals_618, (512, ), (1, ))
    assert_size_stride(primals_619, (512, ), (1, ))
    assert_size_stride(primals_620, (), ())
    assert_size_stride(primals_621, (224, ), (1, ))
    assert_size_stride(primals_622, (224, ), (1, ))
    assert_size_stride(primals_623, (), ())
    assert_size_stride(primals_624, (28, ), (1, ))
    assert_size_stride(primals_625, (28, ), (1, ))
    assert_size_stride(primals_626, (), ())
    assert_size_stride(primals_627, (28, ), (1, ))
    assert_size_stride(primals_628, (28, ), (1, ))
    assert_size_stride(primals_629, (), ())
    assert_size_stride(primals_630, (28, ), (1, ))
    assert_size_stride(primals_631, (28, ), (1, ))
    assert_size_stride(primals_632, (), ())
    assert_size_stride(primals_633, (28, ), (1, ))
    assert_size_stride(primals_634, (28, ), (1, ))
    assert_size_stride(primals_635, (), ())
    assert_size_stride(primals_636, (28, ), (1, ))
    assert_size_stride(primals_637, (28, ), (1, ))
    assert_size_stride(primals_638, (), ())
    assert_size_stride(primals_639, (28, ), (1, ))
    assert_size_stride(primals_640, (28, ), (1, ))
    assert_size_stride(primals_641, (), ())
    assert_size_stride(primals_642, (28, ), (1, ))
    assert_size_stride(primals_643, (28, ), (1, ))
    assert_size_stride(primals_644, (), ())
    assert_size_stride(primals_645, (512, ), (1, ))
    assert_size_stride(primals_646, (512, ), (1, ))
    assert_size_stride(primals_647, (), ())
    assert_size_stride(primals_648, (448, ), (1, ))
    assert_size_stride(primals_649, (448, ), (1, ))
    assert_size_stride(primals_650, (), ())
    assert_size_stride(primals_651, (56, ), (1, ))
    assert_size_stride(primals_652, (56, ), (1, ))
    assert_size_stride(primals_653, (), ())
    assert_size_stride(primals_654, (56, ), (1, ))
    assert_size_stride(primals_655, (56, ), (1, ))
    assert_size_stride(primals_656, (), ())
    assert_size_stride(primals_657, (56, ), (1, ))
    assert_size_stride(primals_658, (56, ), (1, ))
    assert_size_stride(primals_659, (), ())
    assert_size_stride(primals_660, (56, ), (1, ))
    assert_size_stride(primals_661, (56, ), (1, ))
    assert_size_stride(primals_662, (), ())
    assert_size_stride(primals_663, (56, ), (1, ))
    assert_size_stride(primals_664, (56, ), (1, ))
    assert_size_stride(primals_665, (), ())
    assert_size_stride(primals_666, (56, ), (1, ))
    assert_size_stride(primals_667, (56, ), (1, ))
    assert_size_stride(primals_668, (), ())
    assert_size_stride(primals_669, (56, ), (1, ))
    assert_size_stride(primals_670, (56, ), (1, ))
    assert_size_stride(primals_671, (), ())
    assert_size_stride(primals_672, (1024, ), (1, ))
    assert_size_stride(primals_673, (1024, ), (1, ))
    assert_size_stride(primals_674, (), ())
    assert_size_stride(primals_675, (1024, ), (1, ))
    assert_size_stride(primals_676, (1024, ), (1, ))
    assert_size_stride(primals_677, (), ())
    assert_size_stride(primals_678, (448, ), (1, ))
    assert_size_stride(primals_679, (448, ), (1, ))
    assert_size_stride(primals_680, (), ())
    assert_size_stride(primals_681, (56, ), (1, ))
    assert_size_stride(primals_682, (56, ), (1, ))
    assert_size_stride(primals_683, (), ())
    assert_size_stride(primals_684, (56, ), (1, ))
    assert_size_stride(primals_685, (56, ), (1, ))
    assert_size_stride(primals_686, (), ())
    assert_size_stride(primals_687, (56, ), (1, ))
    assert_size_stride(primals_688, (56, ), (1, ))
    assert_size_stride(primals_689, (), ())
    assert_size_stride(primals_690, (56, ), (1, ))
    assert_size_stride(primals_691, (56, ), (1, ))
    assert_size_stride(primals_692, (), ())
    assert_size_stride(primals_693, (56, ), (1, ))
    assert_size_stride(primals_694, (56, ), (1, ))
    assert_size_stride(primals_695, (), ())
    assert_size_stride(primals_696, (56, ), (1, ))
    assert_size_stride(primals_697, (56, ), (1, ))
    assert_size_stride(primals_698, (), ())
    assert_size_stride(primals_699, (56, ), (1, ))
    assert_size_stride(primals_700, (56, ), (1, ))
    assert_size_stride(primals_701, (), ())
    assert_size_stride(primals_702, (1024, ), (1, ))
    assert_size_stride(primals_703, (1024, ), (1, ))
    assert_size_stride(primals_704, (), ())
    assert_size_stride(primals_705, (448, ), (1, ))
    assert_size_stride(primals_706, (448, ), (1, ))
    assert_size_stride(primals_707, (), ())
    assert_size_stride(primals_708, (56, ), (1, ))
    assert_size_stride(primals_709, (56, ), (1, ))
    assert_size_stride(primals_710, (), ())
    assert_size_stride(primals_711, (56, ), (1, ))
    assert_size_stride(primals_712, (56, ), (1, ))
    assert_size_stride(primals_713, (), ())
    assert_size_stride(primals_714, (56, ), (1, ))
    assert_size_stride(primals_715, (56, ), (1, ))
    assert_size_stride(primals_716, (), ())
    assert_size_stride(primals_717, (56, ), (1, ))
    assert_size_stride(primals_718, (56, ), (1, ))
    assert_size_stride(primals_719, (), ())
    assert_size_stride(primals_720, (56, ), (1, ))
    assert_size_stride(primals_721, (56, ), (1, ))
    assert_size_stride(primals_722, (), ())
    assert_size_stride(primals_723, (56, ), (1, ))
    assert_size_stride(primals_724, (56, ), (1, ))
    assert_size_stride(primals_725, (), ())
    assert_size_stride(primals_726, (56, ), (1, ))
    assert_size_stride(primals_727, (56, ), (1, ))
    assert_size_stride(primals_728, (), ())
    assert_size_stride(primals_729, (1024, ), (1, ))
    assert_size_stride(primals_730, (1024, ), (1, ))
    assert_size_stride(primals_731, (), ())
    assert_size_stride(primals_732, (448, ), (1, ))
    assert_size_stride(primals_733, (448, ), (1, ))
    assert_size_stride(primals_734, (), ())
    assert_size_stride(primals_735, (56, ), (1, ))
    assert_size_stride(primals_736, (56, ), (1, ))
    assert_size_stride(primals_737, (), ())
    assert_size_stride(primals_738, (56, ), (1, ))
    assert_size_stride(primals_739, (56, ), (1, ))
    assert_size_stride(primals_740, (), ())
    assert_size_stride(primals_741, (56, ), (1, ))
    assert_size_stride(primals_742, (56, ), (1, ))
    assert_size_stride(primals_743, (), ())
    assert_size_stride(primals_744, (56, ), (1, ))
    assert_size_stride(primals_745, (56, ), (1, ))
    assert_size_stride(primals_746, (), ())
    assert_size_stride(primals_747, (56, ), (1, ))
    assert_size_stride(primals_748, (56, ), (1, ))
    assert_size_stride(primals_749, (), ())
    assert_size_stride(primals_750, (56, ), (1, ))
    assert_size_stride(primals_751, (56, ), (1, ))
    assert_size_stride(primals_752, (), ())
    assert_size_stride(primals_753, (56, ), (1, ))
    assert_size_stride(primals_754, (56, ), (1, ))
    assert_size_stride(primals_755, (), ())
    assert_size_stride(primals_756, (1024, ), (1, ))
    assert_size_stride(primals_757, (1024, ), (1, ))
    assert_size_stride(primals_758, (), ())
    assert_size_stride(primals_759, (448, ), (1, ))
    assert_size_stride(primals_760, (448, ), (1, ))
    assert_size_stride(primals_761, (), ())
    assert_size_stride(primals_762, (56, ), (1, ))
    assert_size_stride(primals_763, (56, ), (1, ))
    assert_size_stride(primals_764, (), ())
    assert_size_stride(primals_765, (56, ), (1, ))
    assert_size_stride(primals_766, (56, ), (1, ))
    assert_size_stride(primals_767, (), ())
    assert_size_stride(primals_768, (56, ), (1, ))
    assert_size_stride(primals_769, (56, ), (1, ))
    assert_size_stride(primals_770, (), ())
    assert_size_stride(primals_771, (56, ), (1, ))
    assert_size_stride(primals_772, (56, ), (1, ))
    assert_size_stride(primals_773, (), ())
    assert_size_stride(primals_774, (56, ), (1, ))
    assert_size_stride(primals_775, (56, ), (1, ))
    assert_size_stride(primals_776, (), ())
    assert_size_stride(primals_777, (56, ), (1, ))
    assert_size_stride(primals_778, (56, ), (1, ))
    assert_size_stride(primals_779, (), ())
    assert_size_stride(primals_780, (56, ), (1, ))
    assert_size_stride(primals_781, (56, ), (1, ))
    assert_size_stride(primals_782, (), ())
    assert_size_stride(primals_783, (1024, ), (1, ))
    assert_size_stride(primals_784, (1024, ), (1, ))
    assert_size_stride(primals_785, (), ())
    assert_size_stride(primals_786, (448, ), (1, ))
    assert_size_stride(primals_787, (448, ), (1, ))
    assert_size_stride(primals_788, (), ())
    assert_size_stride(primals_789, (56, ), (1, ))
    assert_size_stride(primals_790, (56, ), (1, ))
    assert_size_stride(primals_791, (), ())
    assert_size_stride(primals_792, (56, ), (1, ))
    assert_size_stride(primals_793, (56, ), (1, ))
    assert_size_stride(primals_794, (), ())
    assert_size_stride(primals_795, (56, ), (1, ))
    assert_size_stride(primals_796, (56, ), (1, ))
    assert_size_stride(primals_797, (), ())
    assert_size_stride(primals_798, (56, ), (1, ))
    assert_size_stride(primals_799, (56, ), (1, ))
    assert_size_stride(primals_800, (), ())
    assert_size_stride(primals_801, (56, ), (1, ))
    assert_size_stride(primals_802, (56, ), (1, ))
    assert_size_stride(primals_803, (), ())
    assert_size_stride(primals_804, (56, ), (1, ))
    assert_size_stride(primals_805, (56, ), (1, ))
    assert_size_stride(primals_806, (), ())
    assert_size_stride(primals_807, (56, ), (1, ))
    assert_size_stride(primals_808, (56, ), (1, ))
    assert_size_stride(primals_809, (), ())
    assert_size_stride(primals_810, (1024, ), (1, ))
    assert_size_stride(primals_811, (1024, ), (1, ))
    assert_size_stride(primals_812, (), ())
    assert_size_stride(primals_813, (896, ), (1, ))
    assert_size_stride(primals_814, (896, ), (1, ))
    assert_size_stride(primals_815, (), ())
    assert_size_stride(primals_816, (112, ), (1, ))
    assert_size_stride(primals_817, (112, ), (1, ))
    assert_size_stride(primals_818, (), ())
    assert_size_stride(primals_819, (112, ), (1, ))
    assert_size_stride(primals_820, (112, ), (1, ))
    assert_size_stride(primals_821, (), ())
    assert_size_stride(primals_822, (112, ), (1, ))
    assert_size_stride(primals_823, (112, ), (1, ))
    assert_size_stride(primals_824, (), ())
    assert_size_stride(primals_825, (112, ), (1, ))
    assert_size_stride(primals_826, (112, ), (1, ))
    assert_size_stride(primals_827, (), ())
    assert_size_stride(primals_828, (112, ), (1, ))
    assert_size_stride(primals_829, (112, ), (1, ))
    assert_size_stride(primals_830, (), ())
    assert_size_stride(primals_831, (112, ), (1, ))
    assert_size_stride(primals_832, (112, ), (1, ))
    assert_size_stride(primals_833, (), ())
    assert_size_stride(primals_834, (112, ), (1, ))
    assert_size_stride(primals_835, (112, ), (1, ))
    assert_size_stride(primals_836, (), ())
    assert_size_stride(primals_837, (2048, ), (1, ))
    assert_size_stride(primals_838, (2048, ), (1, ))
    assert_size_stride(primals_839, (), ())
    assert_size_stride(primals_840, (2048, ), (1, ))
    assert_size_stride(primals_841, (2048, ), (1, ))
    assert_size_stride(primals_842, (), ())
    assert_size_stride(primals_843, (896, ), (1, ))
    assert_size_stride(primals_844, (896, ), (1, ))
    assert_size_stride(primals_845, (), ())
    assert_size_stride(primals_846, (112, ), (1, ))
    assert_size_stride(primals_847, (112, ), (1, ))
    assert_size_stride(primals_848, (), ())
    assert_size_stride(primals_849, (112, ), (1, ))
    assert_size_stride(primals_850, (112, ), (1, ))
    assert_size_stride(primals_851, (), ())
    assert_size_stride(primals_852, (112, ), (1, ))
    assert_size_stride(primals_853, (112, ), (1, ))
    assert_size_stride(primals_854, (), ())
    assert_size_stride(primals_855, (112, ), (1, ))
    assert_size_stride(primals_856, (112, ), (1, ))
    assert_size_stride(primals_857, (), ())
    assert_size_stride(primals_858, (112, ), (1, ))
    assert_size_stride(primals_859, (112, ), (1, ))
    assert_size_stride(primals_860, (), ())
    assert_size_stride(primals_861, (112, ), (1, ))
    assert_size_stride(primals_862, (112, ), (1, ))
    assert_size_stride(primals_863, (), ())
    assert_size_stride(primals_864, (112, ), (1, ))
    assert_size_stride(primals_865, (112, ), (1, ))
    assert_size_stride(primals_866, (), ())
    assert_size_stride(primals_867, (2048, ), (1, ))
    assert_size_stride(primals_868, (2048, ), (1, ))
    assert_size_stride(primals_869, (), ())
    assert_size_stride(primals_870, (896, ), (1, ))
    assert_size_stride(primals_871, (896, ), (1, ))
    assert_size_stride(primals_872, (), ())
    assert_size_stride(primals_873, (112, ), (1, ))
    assert_size_stride(primals_874, (112, ), (1, ))
    assert_size_stride(primals_875, (), ())
    assert_size_stride(primals_876, (112, ), (1, ))
    assert_size_stride(primals_877, (112, ), (1, ))
    assert_size_stride(primals_878, (), ())
    assert_size_stride(primals_879, (112, ), (1, ))
    assert_size_stride(primals_880, (112, ), (1, ))
    assert_size_stride(primals_881, (), ())
    assert_size_stride(primals_882, (112, ), (1, ))
    assert_size_stride(primals_883, (112, ), (1, ))
    assert_size_stride(primals_884, (), ())
    assert_size_stride(primals_885, (112, ), (1, ))
    assert_size_stride(primals_886, (112, ), (1, ))
    assert_size_stride(primals_887, (), ())
    assert_size_stride(primals_888, (112, ), (1, ))
    assert_size_stride(primals_889, (112, ), (1, ))
    assert_size_stride(primals_890, (), ())
    assert_size_stride(primals_891, (112, ), (1, ))
    assert_size_stride(primals_892, (112, ), (1, ))
    assert_size_stride(primals_893, (), ())
    assert_size_stride(primals_894, (2048, ), (1, ))
    assert_size_stride(primals_895, (2048, ), (1, ))
    assert_size_stride(primals_896, (), ())
    assert_size_stride(primals_897, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 192, 49, grid=grid(192, 49), stream=stream0)
        del primals_1
        buf1 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_7, buf1, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_7
        buf2 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_10, buf2, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_10
        buf3 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_13, buf3, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_13
        buf4 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_16, buf4, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_16
        buf5 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_19, buf5, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_19
        buf6 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_22, buf6, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_22
        buf7 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_25, buf7, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_25
        buf8 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_37, buf8, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_37
        buf9 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_40, buf9, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_40
        buf10 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_43, buf10, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_43
        buf11 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_46, buf11, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_46
        buf12 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_49, buf12, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_49
        buf13 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_52, buf13, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_52
        buf14 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_55, buf14, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_55
        buf15 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_64, buf15, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_64
        buf16 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_67, buf16, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_67
        buf17 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_70, buf17, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_70
        buf18 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_73, buf18, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_73
        buf19 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_76, buf19, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_76
        buf20 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_79, buf20, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_79
        buf21 = empty_strided((14, 14, 3, 3), (126, 1, 42, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_82, buf21, 196, 9, grid=grid(196, 9), stream=stream0)
        del primals_82
        buf22 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_91, buf22, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_91
        buf23 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_94, buf23, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_94
        buf24 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_97, buf24, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_97
        buf25 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_100, buf25, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_100
        buf26 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_103, buf26, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_103
        buf27 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_106, buf27, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_106
        buf28 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_109, buf28, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_109
        buf29 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_121, buf29, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_121
        buf30 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_124, buf30, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_124
        buf31 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_127, buf31, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_127
        buf32 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_130, buf32, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_130
        buf33 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_133, buf33, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_133
        buf34 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_136, buf34, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_136
        buf35 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_139, buf35, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_139
        buf36 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_148, buf36, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_148
        buf37 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_151, buf37, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_151
        buf38 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_154, buf38, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_154
        buf39 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_157, buf39, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_157
        buf40 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_160, buf40, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_160
        buf41 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_163, buf41, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_163
        buf42 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_166, buf42, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_166
        buf43 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_175, buf43, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_175
        buf44 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_178, buf44, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_178
        buf45 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_181, buf45, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_181
        buf46 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_184, buf46, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_184
        buf47 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_187, buf47, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_187
        buf48 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_190, buf48, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_190
        buf49 = empty_strided((28, 28, 3, 3), (252, 1, 84, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_193, buf49, 784, 9, grid=grid(784, 9), stream=stream0)
        del primals_193
        buf50 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_202, buf50, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_202
        buf51 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_205, buf51, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_205
        buf52 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_208, buf52, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_208
        buf53 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_211, buf53, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_211
        buf54 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_214, buf54, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_214
        buf55 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_217, buf55, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_217
        buf56 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_220, buf56, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_220
        buf57 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_232, buf57, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_232
        buf58 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_235, buf58, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_235
        buf59 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_238, buf59, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_238
        buf60 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_241, buf60, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_241
        buf61 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_244, buf61, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_244
        buf62 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_247, buf62, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_247
        buf63 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_250, buf63, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_250
        buf64 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_259, buf64, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_259
        buf65 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_262, buf65, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_262
        buf66 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_265, buf66, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_265
        buf67 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_268, buf67, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_268
        buf68 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_271, buf68, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_271
        buf69 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_274, buf69, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_274
        buf70 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_277, buf70, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_277
        buf71 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_286, buf71, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_286
        buf72 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_289, buf72, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_289
        buf73 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_292, buf73, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_292
        buf74 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_295, buf74, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_295
        buf75 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_298, buf75, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_298
        buf76 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_301, buf76, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_301
        buf77 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_304, buf77, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_304
        buf78 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_313, buf78, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_313
        buf79 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_316, buf79, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_316
        buf80 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_319, buf80, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_319
        buf81 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_322, buf81, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_322
        buf82 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_325, buf82, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_325
        buf83 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_328, buf83, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_328
        buf84 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_331, buf84, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_331
        buf85 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_340, buf85, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_340
        buf86 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_343, buf86, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_343
        buf87 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_346, buf87, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_346
        buf88 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_349, buf88, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_349
        buf89 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_352, buf89, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_352
        buf90 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_355, buf90, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_355
        buf91 = empty_strided((56, 56, 3, 3), (504, 1, 168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_358, buf91, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del primals_358
        buf92 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_367, buf92, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_367
        buf93 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_370, buf93, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_370
        buf94 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_373, buf94, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_373
        buf95 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_376, buf95, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_376
        buf96 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_379, buf96, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_379
        buf97 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_382, buf97, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_382
        buf98 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_385, buf98, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_385
        buf99 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_397, buf99, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_397
        buf100 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_400, buf100, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_400
        buf101 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_403, buf101, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_403
        buf102 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_406, buf102, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_406
        buf103 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_409, buf103, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_409
        buf104 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_412, buf104, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_412
        buf105 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_415, buf105, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_415
        buf106 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_424, buf106, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_424
        buf107 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_427, buf107, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_427
        buf108 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_430, buf108, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_430
        buf109 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_433, buf109, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_433
        buf110 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_436, buf110, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_436
        buf111 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_439, buf111, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_439
        buf112 = empty_strided((112, 112, 3, 3), (1008, 1, 336, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_442, buf112, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del primals_442
        buf113 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_897, buf113, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del primals_897
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, buf0, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 64, 112, 112), (802816, 12544, 112, 1))
        buf115 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(buf114, buf115, 512, 12544, grid=grid(512, 12544), stream=stream0)
        buf116 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        buf117 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        buf118 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf115, buf116, buf117, buf118, 50176, 128, grid=grid(50176), stream=stream0)
        buf119 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        buf120 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        buf121 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf116, buf117, buf118, buf119, buf120, buf121, 448, 112, grid=grid(448), stream=stream0)
        buf122 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf123 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf125 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_9.run(buf119, buf120, buf121, primals_450, primals_451, buf122, buf123, buf125, primals_450, primals_451, 64, 7, grid=grid(64), stream=stream0)
        del primals_450
        del primals_451
        buf126 = reinterpret_tensor(buf114, (8, 64, 112, 112), (802816, 1, 7168, 64), 0); del buf114  # reuse
        # Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_10.run(buf115, buf122, buf123, primals_2, primals_3, buf126, 6422528, grid=grid(6422528), stream=stream0)
        del buf123
        del primals_3
        buf127 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        buf128 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.int64)
        # Source Nodes: [shortcut], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_11.run(buf126, buf127, buf128, 1605632, grid=grid(1605632), stream=stream0)
        # Source Nodes: [out], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf127, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (8, 112, 56, 56), (351232, 3136, 56, 1))
        buf130 = empty_strided((8, 112, 56, 56), (351232, 1, 6272, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [out], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf129, buf130, 896, 3136, grid=grid(896, 3136), stream=stream0)
        buf131 = empty_strided((1, 112, 1, 1, 196), (21952, 1, 21952, 21952, 112), device='cuda', dtype=torch.float32)
        buf132 = empty_strided((1, 112, 1, 1, 196), (21952, 1, 21952, 21952, 112), device='cuda', dtype=torch.float32)
        buf133 = empty_strided((1, 112, 1, 1, 196), (21952, 1, 21952, 21952, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf130, buf131, buf132, buf133, 21952, 128, grid=grid(21952), stream=stream0)
        buf134 = empty_strided((1, 112, 1, 1, 2), (224, 1, 224, 224, 112), device='cuda', dtype=torch.float32)
        buf135 = empty_strided((1, 112, 1, 1, 2), (224, 1, 224, 224, 112), device='cuda', dtype=torch.float32)
        buf136 = empty_strided((1, 112, 1, 1, 2), (224, 1, 224, 224, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf131, buf132, buf133, buf134, buf135, buf136, 224, 98, grid=grid(224), stream=stream0)
        buf137 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf138 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf140 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf134, buf135, buf136, primals_453, primals_454, buf137, buf138, buf140, primals_453, primals_454, 112, 2, grid=grid(112), stream=stream0)
        del primals_453
        del primals_454
        buf141 = reinterpret_tensor(buf129, (8, 112, 56, 56), (351232, 1, 6272, 112), 0); del buf129  # reuse
        buf1947 = empty_strided((8, 112, 56, 56), (351232, 1, 6272, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_16.run(buf130, buf137, buf138, primals_5, primals_6, buf141, buf1947, 2809856, grid=grid(2809856), stream=stream0)
        del primals_6
        # Source Nodes: [sp_1], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(reinterpret_tensor(buf141, (8, 14, 56, 56), (351232, 1, 6272, 112), 0), buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf143 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf142, buf143, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf144 = empty_strided((1, 14, 1, 1, 196), (2744, 1, 2744, 2744, 14), device='cuda', dtype=torch.float32)
        buf145 = empty_strided((1, 14, 1, 1, 196), (2744, 1, 2744, 2744, 14), device='cuda', dtype=torch.float32)
        buf146 = empty_strided((1, 14, 1, 1, 196), (2744, 1, 2744, 2744, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_2], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf143, buf144, buf145, buf146, 2744, 128, grid=grid(2744), stream=stream0)
        buf147 = empty_strided((1, 14, 1, 1, 2), (28, 1, 28, 28, 14), device='cuda', dtype=torch.float32)
        buf148 = empty_strided((1, 14, 1, 1, 2), (28, 1, 28, 28, 14), device='cuda', dtype=torch.float32)
        buf149 = empty_strided((1, 14, 1, 1, 2), (28, 1, 28, 28, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_2], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf144, buf145, buf146, buf147, buf148, buf149, 28, 98, grid=grid(28), stream=stream0)
        buf150 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf151 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf153 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_2], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf147, buf148, buf149, primals_456, primals_457, buf150, buf151, buf153, primals_456, primals_457, 14, 2, grid=grid(14), stream=stream0)
        del primals_456
        del primals_457
        buf234 = empty((8, 112, 56, 56), device='cuda', dtype=torch.float32)
        buf154 = reinterpret_tensor(buf234, (8, 14, 56, 56), (351232, 3136, 56, 1), 0)  # alias
        buf1946 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_2, sp_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf143, buf150, buf151, primals_8, primals_9, buf154, buf1946, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_9
        # Source Nodes: [sp_5], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(reinterpret_tensor(buf141, (8, 14, 56, 56), (351232, 1, 6272, 112), 14), buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf156 = reinterpret_tensor(buf142, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf142  # reuse
        # Source Nodes: [sp_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf155, buf156, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf157 = buf146; del buf146  # reuse
        buf158 = buf145; del buf145  # reuse
        buf159 = buf144; del buf144  # reuse
        # Source Nodes: [sp_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf156, buf157, buf158, buf159, 2744, 128, grid=grid(2744), stream=stream0)
        buf160 = buf149; del buf149  # reuse
        buf161 = buf148; del buf148  # reuse
        buf162 = buf147; del buf147  # reuse
        # Source Nodes: [sp_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf157, buf158, buf159, buf160, buf161, buf162, 28, 98, grid=grid(28), stream=stream0)
        buf163 = buf151; del buf151  # reuse
        buf164 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf166 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf160, buf161, buf162, primals_459, primals_460, buf163, buf164, buf166, primals_459, primals_460, 14, 2, grid=grid(14), stream=stream0)
        del primals_459
        del primals_460
        buf167 = reinterpret_tensor(buf234, (8, 14, 56, 56), (351232, 3136, 56, 1), 43904)  # alias
        buf1945 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_6, sp_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf156, buf163, buf164, primals_11, primals_12, buf167, buf1945, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_12
        # Source Nodes: [sp_9], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(reinterpret_tensor(buf141, (8, 14, 56, 56), (351232, 1, 6272, 112), 28), buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf169 = reinterpret_tensor(buf155, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf155  # reuse
        # Source Nodes: [sp_9], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf168, buf169, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf170 = buf159; del buf159  # reuse
        buf171 = buf158; del buf158  # reuse
        buf172 = buf157; del buf157  # reuse
        # Source Nodes: [sp_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf169, buf170, buf171, buf172, 2744, 128, grid=grid(2744), stream=stream0)
        buf173 = buf162; del buf162  # reuse
        buf174 = buf161; del buf161  # reuse
        buf175 = buf160; del buf160  # reuse
        # Source Nodes: [sp_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf170, buf171, buf172, buf173, buf174, buf175, 28, 98, grid=grid(28), stream=stream0)
        buf176 = buf164; del buf164  # reuse
        buf177 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf179 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf173, buf174, buf175, primals_462, primals_463, buf176, buf177, buf179, primals_462, primals_463, 14, 2, grid=grid(14), stream=stream0)
        del primals_462
        del primals_463
        buf180 = reinterpret_tensor(buf234, (8, 14, 56, 56), (351232, 3136, 56, 1), 87808)  # alias
        buf1944 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_10, sp_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf169, buf176, buf177, primals_14, primals_15, buf180, buf1944, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_15
        # Source Nodes: [sp_13], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(reinterpret_tensor(buf141, (8, 14, 56, 56), (351232, 1, 6272, 112), 42), buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf182 = reinterpret_tensor(buf168, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf168  # reuse
        # Source Nodes: [sp_13], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf181, buf182, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf183 = buf172; del buf172  # reuse
        buf184 = buf171; del buf171  # reuse
        buf185 = buf170; del buf170  # reuse
        # Source Nodes: [sp_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf182, buf183, buf184, buf185, 2744, 128, grid=grid(2744), stream=stream0)
        buf186 = buf175; del buf175  # reuse
        buf187 = buf174; del buf174  # reuse
        buf188 = buf173; del buf173  # reuse
        # Source Nodes: [sp_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf183, buf184, buf185, buf186, buf187, buf188, 28, 98, grid=grid(28), stream=stream0)
        buf189 = buf177; del buf177  # reuse
        buf190 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf192 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf186, buf187, buf188, primals_465, primals_466, buf189, buf190, buf192, primals_465, primals_466, 14, 2, grid=grid(14), stream=stream0)
        del primals_465
        del primals_466
        buf193 = reinterpret_tensor(buf234, (8, 14, 56, 56), (351232, 3136, 56, 1), 131712)  # alias
        buf1943 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_14, sp_15], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf182, buf189, buf190, primals_17, primals_18, buf193, buf1943, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_18
        # Source Nodes: [sp_17], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(reinterpret_tensor(buf141, (8, 14, 56, 56), (351232, 1, 6272, 112), 56), buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf195 = reinterpret_tensor(buf181, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf181  # reuse
        # Source Nodes: [sp_17], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf194, buf195, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf196 = buf185; del buf185  # reuse
        buf197 = buf184; del buf184  # reuse
        buf198 = buf183; del buf183  # reuse
        # Source Nodes: [sp_18], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf195, buf196, buf197, buf198, 2744, 128, grid=grid(2744), stream=stream0)
        buf199 = buf188; del buf188  # reuse
        buf200 = buf187; del buf187  # reuse
        buf201 = buf186; del buf186  # reuse
        # Source Nodes: [sp_18], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf196, buf197, buf198, buf199, buf200, buf201, 28, 98, grid=grid(28), stream=stream0)
        buf202 = buf190; del buf190  # reuse
        buf203 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf205 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_18], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf199, buf200, buf201, primals_468, primals_469, buf202, buf203, buf205, primals_468, primals_469, 14, 2, grid=grid(14), stream=stream0)
        del primals_468
        del primals_469
        buf206 = reinterpret_tensor(buf234, (8, 14, 56, 56), (351232, 3136, 56, 1), 175616)  # alias
        buf1942 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_18, sp_19], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf195, buf202, buf203, primals_20, primals_21, buf206, buf1942, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_21
        # Source Nodes: [sp_21], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(reinterpret_tensor(buf141, (8, 14, 56, 56), (351232, 1, 6272, 112), 70), buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf208 = reinterpret_tensor(buf194, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf194  # reuse
        # Source Nodes: [sp_21], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf207, buf208, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf209 = buf198; del buf198  # reuse
        buf210 = buf197; del buf197  # reuse
        buf211 = buf196; del buf196  # reuse
        # Source Nodes: [sp_22], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf208, buf209, buf210, buf211, 2744, 128, grid=grid(2744), stream=stream0)
        buf212 = buf201; del buf201  # reuse
        buf213 = buf200; del buf200  # reuse
        buf214 = buf199; del buf199  # reuse
        # Source Nodes: [sp_22], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf209, buf210, buf211, buf212, buf213, buf214, 28, 98, grid=grid(28), stream=stream0)
        buf215 = buf203; del buf203  # reuse
        buf216 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf218 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_22], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf212, buf213, buf214, primals_471, primals_472, buf215, buf216, buf218, primals_471, primals_472, 14, 2, grid=grid(14), stream=stream0)
        del primals_471
        del primals_472
        buf219 = reinterpret_tensor(buf234, (8, 14, 56, 56), (351232, 3136, 56, 1), 219520)  # alias
        buf1941 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_22, sp_23], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf208, buf215, buf216, primals_23, primals_24, buf219, buf1941, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_24
        # Source Nodes: [sp_25], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(reinterpret_tensor(buf141, (8, 14, 56, 56), (351232, 1, 6272, 112), 84), buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf221 = reinterpret_tensor(buf207, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf207  # reuse
        # Source Nodes: [sp_25], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf220, buf221, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf222 = buf211; del buf211  # reuse
        buf223 = buf210; del buf210  # reuse
        buf224 = buf209; del buf209  # reuse
        # Source Nodes: [sp_26], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf221, buf222, buf223, buf224, 2744, 128, grid=grid(2744), stream=stream0)
        buf225 = buf214; del buf214  # reuse
        buf226 = buf213; del buf213  # reuse
        buf227 = buf212; del buf212  # reuse
        # Source Nodes: [sp_26], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf222, buf223, buf224, buf225, buf226, buf227, 28, 98, grid=grid(28), stream=stream0)
        buf228 = buf216; del buf216  # reuse
        buf229 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf231 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_26], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf225, buf226, buf227, primals_474, primals_475, buf228, buf229, buf231, primals_474, primals_475, 14, 2, grid=grid(14), stream=stream0)
        del primals_474
        del primals_475
        buf232 = reinterpret_tensor(buf234, (8, 14, 56, 56), (351232, 3136, 56, 1), 263424)  # alias
        buf1940 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_26, sp_27], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf221, buf228, buf229, primals_26, primals_27, buf232, buf1940, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_27
        buf233 = reinterpret_tensor(buf234, (8, 14, 56, 56), (351232, 3136, 56, 1), 307328)  # alias
        # Source Nodes: [getattr_l__mod___layer1___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_22.run(buf141, buf233, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf235 = empty_strided((8, 112, 56, 56), (351232, 1, 6272, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_31], Original ATen: [aten.cat]
        triton_poi_fused_convolution_12.run(buf234, buf235, 896, 3136, grid=grid(896, 3136), stream=stream0)
        del buf154
        del buf167
        del buf180
        del buf193
        del buf206
        del buf219
        del buf232
        del buf233
        # Source Nodes: [out_4], Original ATen: [aten.convolution]
        buf236 = extern_kernels.convolution(buf235, primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf237 = empty_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_4], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf236, buf237, 2048, 3136, grid=grid(2048, 3136), stream=stream0)
        buf238 = reinterpret_tensor(buf118, (1, 256, 1, 1, 196), (50176, 1, 50176, 50176, 256), 0); del buf118  # reuse
        buf239 = reinterpret_tensor(buf117, (1, 256, 1, 1, 196), (50176, 1, 50176, 50176, 256), 0); del buf117  # reuse
        buf240 = reinterpret_tensor(buf116, (1, 256, 1, 1, 196), (50176, 1, 50176, 50176, 256), 0); del buf116  # reuse
        # Source Nodes: [out_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf237, buf238, buf239, buf240, 50176, 128, grid=grid(50176), stream=stream0)
        buf241 = empty_strided((1, 256, 1, 1, 2), (512, 1, 512, 512, 256), device='cuda', dtype=torch.float32)
        buf242 = empty_strided((1, 256, 1, 1, 2), (512, 1, 512, 512, 256), device='cuda', dtype=torch.float32)
        buf243 = empty_strided((1, 256, 1, 1, 2), (512, 1, 512, 512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf238, buf239, buf240, buf241, buf242, buf243, 512, 98, grid=grid(512), stream=stream0)
        buf244 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf245 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf247 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf241, buf242, buf243, primals_477, primals_478, buf244, buf245, buf247, primals_477, primals_478, 256, 2, grid=grid(256), stream=stream0)
        del primals_477
        del primals_478
        # Source Nodes: [getattr_l__mod___layer1___0___downsample_0], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf127, primals_31, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf249 = reinterpret_tensor(buf236, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf236  # reuse
        # Source Nodes: [getattr_l__mod___layer1___0___downsample_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf248, buf249, 2048, 3136, grid=grid(2048, 3136), stream=stream0)
        buf250 = buf240; del buf240  # reuse
        buf251 = buf239; del buf239  # reuse
        buf252 = buf238; del buf238  # reuse
        # Source Nodes: [shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf249, buf250, buf251, buf252, 50176, 128, grid=grid(50176), stream=stream0)
        buf253 = buf243; del buf243  # reuse
        buf254 = buf242; del buf242  # reuse
        buf255 = buf241; del buf241  # reuse
        # Source Nodes: [shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf250, buf251, buf252, buf253, buf254, buf255, 512, 98, grid=grid(512), stream=stream0)
        buf256 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf257 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf259 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf253, buf254, buf255, primals_480, primals_481, buf256, buf257, buf259, primals_480, primals_481, 256, 2, grid=grid(256), stream=stream0)
        del primals_480
        del primals_481
        buf260 = reinterpret_tensor(buf248, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf248  # reuse
        buf261 = buf260; del buf260  # reuse
        # Source Nodes: [out_5, out_6, shortcut_1, shortcut_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_27.run(buf261, buf237, buf244, buf245, primals_29, primals_30, buf249, buf256, buf257, primals_32, primals_33, 6422528, grid=grid(6422528), stream=stream0)
        del primals_30
        del primals_33
        # Source Nodes: [out_8], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf261, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (8, 112, 56, 56), (351232, 3136, 56, 1))
        buf263 = reinterpret_tensor(buf234, (8, 112, 56, 56), (351232, 1, 6272, 112), 0); del buf234  # reuse
        # Source Nodes: [out_8], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf262, buf263, 896, 3136, grid=grid(896, 3136), stream=stream0)
        buf264 = buf133; del buf133  # reuse
        buf265 = buf132; del buf132  # reuse
        buf266 = buf131; del buf131  # reuse
        # Source Nodes: [out_9], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf263, buf264, buf265, buf266, 21952, 128, grid=grid(21952), stream=stream0)
        buf267 = buf136; del buf136  # reuse
        buf268 = buf135; del buf135  # reuse
        buf269 = buf134; del buf134  # reuse
        # Source Nodes: [out_9], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf264, buf265, buf266, buf267, buf268, buf269, 224, 98, grid=grid(224), stream=stream0)
        buf270 = buf138; del buf138  # reuse
        buf271 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf273 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_9], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf267, buf268, buf269, primals_483, primals_484, buf270, buf271, buf273, primals_483, primals_484, 112, 2, grid=grid(112), stream=stream0)
        del primals_483
        del primals_484
        buf274 = reinterpret_tensor(buf262, (8, 112, 56, 56), (351232, 1, 6272, 112), 0); del buf262  # reuse
        buf1939 = empty_strided((8, 112, 56, 56), (351232, 1, 6272, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_10, out_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_16.run(buf263, buf270, buf271, primals_35, primals_36, buf274, buf1939, 2809856, grid=grid(2809856), stream=stream0)
        del primals_36
        # Source Nodes: [sp_30], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(reinterpret_tensor(buf274, (8, 14, 56, 56), (351232, 1, 6272, 112), 0), buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf276 = reinterpret_tensor(buf220, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf220  # reuse
        # Source Nodes: [sp_30], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf275, buf276, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf277 = buf224; del buf224  # reuse
        buf278 = buf223; del buf223  # reuse
        buf279 = buf222; del buf222  # reuse
        # Source Nodes: [sp_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf276, buf277, buf278, buf279, 2744, 128, grid=grid(2744), stream=stream0)
        buf280 = buf227; del buf227  # reuse
        buf281 = buf226; del buf226  # reuse
        buf282 = buf225; del buf225  # reuse
        # Source Nodes: [sp_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf277, buf278, buf279, buf280, buf281, buf282, 28, 98, grid=grid(28), stream=stream0)
        buf283 = buf229; del buf229  # reuse
        buf284 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf286 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf280, buf281, buf282, primals_486, primals_487, buf283, buf284, buf286, primals_486, primals_487, 14, 2, grid=grid(14), stream=stream0)
        del primals_486
        del primals_487
        buf373 = empty((8, 112, 56, 56), device='cuda', dtype=torch.float32)
        buf287 = reinterpret_tensor(buf373, (8, 14, 56, 56), (351232, 3136, 56, 1), 0)  # alias
        buf1938 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_31, sp_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf276, buf283, buf284, primals_38, primals_39, buf287, buf1938, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_39
        buf288 = reinterpret_tensor(buf275, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf275  # reuse
        # Source Nodes: [sp_33], Original ATen: [aten.add]
        triton_poi_fused_add_28.run(buf287, buf274, buf288, 25088, 14, grid=grid(25088, 14), stream=stream0)
        # Source Nodes: [sp_34], Original ATen: [aten.convolution]
        buf289 = extern_kernels.convolution(buf288, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf289, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf290 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_34], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf289, buf290, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf291 = buf279; del buf279  # reuse
        buf292 = buf278; del buf278  # reuse
        buf293 = buf277; del buf277  # reuse
        # Source Nodes: [sp_35], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf290, buf291, buf292, buf293, 2744, 128, grid=grid(2744), stream=stream0)
        buf294 = buf282; del buf282  # reuse
        buf295 = buf281; del buf281  # reuse
        buf296 = buf280; del buf280  # reuse
        # Source Nodes: [sp_35], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf291, buf292, buf293, buf294, buf295, buf296, 28, 98, grid=grid(28), stream=stream0)
        buf297 = buf284; del buf284  # reuse
        buf298 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf300 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_35], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf294, buf295, buf296, primals_489, primals_490, buf297, buf298, buf300, primals_489, primals_490, 14, 2, grid=grid(14), stream=stream0)
        del primals_489
        del primals_490
        buf301 = reinterpret_tensor(buf373, (8, 14, 56, 56), (351232, 3136, 56, 1), 43904)  # alias
        buf1937 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_35, sp_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf290, buf297, buf298, primals_41, primals_42, buf301, buf1937, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_42
        buf302 = reinterpret_tensor(buf289, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf289  # reuse
        # Source Nodes: [sp_37], Original ATen: [aten.add]
        triton_poi_fused_add_29.run(buf301, buf274, buf302, 25088, 14, grid=grid(25088, 14), stream=stream0)
        # Source Nodes: [sp_38], Original ATen: [aten.convolution]
        buf303 = extern_kernels.convolution(buf302, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf303, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf304 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_38], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf303, buf304, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf305 = buf293; del buf293  # reuse
        buf306 = buf292; del buf292  # reuse
        buf307 = buf291; del buf291  # reuse
        # Source Nodes: [sp_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf304, buf305, buf306, buf307, 2744, 128, grid=grid(2744), stream=stream0)
        buf308 = buf296; del buf296  # reuse
        buf309 = buf295; del buf295  # reuse
        buf310 = buf294; del buf294  # reuse
        # Source Nodes: [sp_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf305, buf306, buf307, buf308, buf309, buf310, 28, 98, grid=grid(28), stream=stream0)
        buf311 = buf298; del buf298  # reuse
        buf312 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf314 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf308, buf309, buf310, primals_492, primals_493, buf311, buf312, buf314, primals_492, primals_493, 14, 2, grid=grid(14), stream=stream0)
        del primals_492
        del primals_493
        buf315 = reinterpret_tensor(buf373, (8, 14, 56, 56), (351232, 3136, 56, 1), 87808)  # alias
        buf1936 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_39, sp_40], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf304, buf311, buf312, primals_44, primals_45, buf315, buf1936, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_45
        buf316 = reinterpret_tensor(buf303, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf303  # reuse
        # Source Nodes: [sp_41], Original ATen: [aten.add]
        triton_poi_fused_add_30.run(buf315, buf274, buf316, 25088, 14, grid=grid(25088, 14), stream=stream0)
        # Source Nodes: [sp_42], Original ATen: [aten.convolution]
        buf317 = extern_kernels.convolution(buf316, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf317, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf318 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_42], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf317, buf318, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf319 = buf307; del buf307  # reuse
        buf320 = buf306; del buf306  # reuse
        buf321 = buf305; del buf305  # reuse
        # Source Nodes: [sp_43], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf318, buf319, buf320, buf321, 2744, 128, grid=grid(2744), stream=stream0)
        buf322 = buf310; del buf310  # reuse
        buf323 = buf309; del buf309  # reuse
        buf324 = buf308; del buf308  # reuse
        # Source Nodes: [sp_43], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf319, buf320, buf321, buf322, buf323, buf324, 28, 98, grid=grid(28), stream=stream0)
        buf325 = buf312; del buf312  # reuse
        buf326 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf328 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_43], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf322, buf323, buf324, primals_495, primals_496, buf325, buf326, buf328, primals_495, primals_496, 14, 2, grid=grid(14), stream=stream0)
        del primals_495
        del primals_496
        buf329 = reinterpret_tensor(buf373, (8, 14, 56, 56), (351232, 3136, 56, 1), 131712)  # alias
        buf1935 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_43, sp_44], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf318, buf325, buf326, primals_47, primals_48, buf329, buf1935, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_48
        buf330 = reinterpret_tensor(buf317, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf317  # reuse
        # Source Nodes: [sp_45], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(buf329, buf274, buf330, 25088, 14, grid=grid(25088, 14), stream=stream0)
        # Source Nodes: [sp_46], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf330, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf332 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_46], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf331, buf332, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf333 = buf321; del buf321  # reuse
        buf334 = buf320; del buf320  # reuse
        buf335 = buf319; del buf319  # reuse
        # Source Nodes: [sp_47], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf332, buf333, buf334, buf335, 2744, 128, grid=grid(2744), stream=stream0)
        buf336 = buf324; del buf324  # reuse
        buf337 = buf323; del buf323  # reuse
        buf338 = buf322; del buf322  # reuse
        # Source Nodes: [sp_47], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf333, buf334, buf335, buf336, buf337, buf338, 28, 98, grid=grid(28), stream=stream0)
        buf339 = buf326; del buf326  # reuse
        buf340 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf342 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_47], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf336, buf337, buf338, primals_498, primals_499, buf339, buf340, buf342, primals_498, primals_499, 14, 2, grid=grid(14), stream=stream0)
        del primals_498
        del primals_499
        buf343 = reinterpret_tensor(buf373, (8, 14, 56, 56), (351232, 3136, 56, 1), 175616)  # alias
        buf1934 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_47, sp_48], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf332, buf339, buf340, primals_50, primals_51, buf343, buf1934, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_51
        buf344 = reinterpret_tensor(buf331, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf331  # reuse
        # Source Nodes: [sp_49], Original ATen: [aten.add]
        triton_poi_fused_add_32.run(buf343, buf274, buf344, 25088, 14, grid=grid(25088, 14), stream=stream0)
        # Source Nodes: [sp_50], Original ATen: [aten.convolution]
        buf345 = extern_kernels.convolution(buf344, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf345, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf346 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_50], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf345, buf346, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf347 = buf335; del buf335  # reuse
        buf348 = buf334; del buf334  # reuse
        buf349 = buf333; del buf333  # reuse
        # Source Nodes: [sp_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf346, buf347, buf348, buf349, 2744, 128, grid=grid(2744), stream=stream0)
        buf350 = buf338; del buf338  # reuse
        buf351 = buf337; del buf337  # reuse
        buf352 = buf336; del buf336  # reuse
        # Source Nodes: [sp_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf347, buf348, buf349, buf350, buf351, buf352, 28, 98, grid=grid(28), stream=stream0)
        buf353 = buf340; del buf340  # reuse
        buf354 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf356 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf350, buf351, buf352, primals_501, primals_502, buf353, buf354, buf356, primals_501, primals_502, 14, 2, grid=grid(14), stream=stream0)
        del primals_501
        del primals_502
        buf357 = reinterpret_tensor(buf373, (8, 14, 56, 56), (351232, 3136, 56, 1), 219520)  # alias
        buf1933 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_51, sp_52], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf346, buf353, buf354, primals_53, primals_54, buf357, buf1933, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_54
        buf358 = reinterpret_tensor(buf345, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf345  # reuse
        # Source Nodes: [sp_53], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf357, buf274, buf358, 25088, 14, grid=grid(25088, 14), stream=stream0)
        # Source Nodes: [sp_54], Original ATen: [aten.convolution]
        buf359 = extern_kernels.convolution(buf358, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf359, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf360 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_54], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf359, buf360, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf361 = buf349; del buf349  # reuse
        buf362 = buf348; del buf348  # reuse
        buf363 = buf347; del buf347  # reuse
        # Source Nodes: [sp_55], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf360, buf361, buf362, buf363, 2744, 128, grid=grid(2744), stream=stream0)
        buf364 = buf352; del buf352  # reuse
        buf365 = buf351; del buf351  # reuse
        buf366 = buf350; del buf350  # reuse
        # Source Nodes: [sp_55], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf361, buf362, buf363, buf364, buf365, buf366, 28, 98, grid=grid(28), stream=stream0)
        buf367 = buf354; del buf354  # reuse
        buf368 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf370 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_55], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf364, buf365, buf366, primals_504, primals_505, buf367, buf368, buf370, primals_504, primals_505, 14, 2, grid=grid(14), stream=stream0)
        del primals_504
        del primals_505
        buf371 = reinterpret_tensor(buf373, (8, 14, 56, 56), (351232, 3136, 56, 1), 263424)  # alias
        buf1932 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_55, sp_56], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf360, buf367, buf368, primals_56, primals_57, buf371, buf1932, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_57
        buf372 = reinterpret_tensor(buf373, (8, 14, 56, 56), (351232, 3136, 56, 1), 307328)  # alias
        # Source Nodes: [cat_30], Original ATen: [aten.cat]
        triton_poi_fused_cat_34.run(buf274, buf372, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf374 = empty_strided((8, 112, 56, 56), (351232, 1, 6272, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_30], Original ATen: [aten.cat]
        triton_poi_fused_convolution_12.run(buf373, buf374, 896, 3136, grid=grid(896, 3136), stream=stream0)
        del buf287
        del buf301
        del buf315
        del buf329
        del buf343
        del buf357
        del buf371
        del buf372
        # Source Nodes: [out_12], Original ATen: [aten.convolution]
        buf375 = extern_kernels.convolution(buf374, primals_58, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf375, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf376 = empty_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_12], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf375, buf376, 2048, 3136, grid=grid(2048, 3136), stream=stream0)
        buf377 = buf252; del buf252  # reuse
        buf378 = buf251; del buf251  # reuse
        buf379 = buf250; del buf250  # reuse
        # Source Nodes: [out_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf376, buf377, buf378, buf379, 50176, 128, grid=grid(50176), stream=stream0)
        buf380 = buf255; del buf255  # reuse
        buf381 = buf254; del buf254  # reuse
        buf382 = buf253; del buf253  # reuse
        # Source Nodes: [out_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf377, buf378, buf379, buf380, buf381, buf382, 512, 98, grid=grid(512), stream=stream0)
        buf383 = buf257; del buf257  # reuse
        buf384 = buf245; del buf245  # reuse
        buf386 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf380, buf381, buf382, primals_507, primals_508, buf383, buf384, buf386, primals_507, primals_508, 256, 2, grid=grid(256), stream=stream0)
        del primals_507
        del primals_508
        buf387 = reinterpret_tensor(buf375, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf375  # reuse
        # Source Nodes: [out_13, out_14, shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_35.run(buf376, buf383, buf384, primals_59, primals_60, buf261, buf387, 6422528, grid=grid(6422528), stream=stream0)
        del primals_60
        # Source Nodes: [out_16], Original ATen: [aten.convolution]
        buf388 = extern_kernels.convolution(buf387, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf388, (8, 112, 56, 56), (351232, 3136, 56, 1))
        buf389 = reinterpret_tensor(buf373, (8, 112, 56, 56), (351232, 1, 6272, 112), 0); del buf373  # reuse
        # Source Nodes: [out_16], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf388, buf389, 896, 3136, grid=grid(896, 3136), stream=stream0)
        buf390 = buf266; del buf266  # reuse
        buf391 = buf265; del buf265  # reuse
        buf392 = buf264; del buf264  # reuse
        # Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf389, buf390, buf391, buf392, 21952, 128, grid=grid(21952), stream=stream0)
        buf393 = buf269; del buf269  # reuse
        buf394 = buf268; del buf268  # reuse
        buf395 = buf267; del buf267  # reuse
        # Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf390, buf391, buf392, buf393, buf394, buf395, 224, 98, grid=grid(224), stream=stream0)
        buf396 = buf271; del buf271  # reuse
        buf397 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf399 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf393, buf394, buf395, primals_510, primals_511, buf396, buf397, buf399, primals_510, primals_511, 112, 2, grid=grid(112), stream=stream0)
        del primals_510
        del primals_511
        buf400 = reinterpret_tensor(buf388, (8, 112, 56, 56), (351232, 1, 6272, 112), 0); del buf388  # reuse
        buf1931 = empty_strided((8, 112, 56, 56), (351232, 1, 6272, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_17, out_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_16.run(buf389, buf396, buf397, primals_62, primals_63, buf400, buf1931, 2809856, grid=grid(2809856), stream=stream0)
        del primals_63
        # Source Nodes: [sp_59], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(reinterpret_tensor(buf400, (8, 14, 56, 56), (351232, 1, 6272, 112), 0), buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf402 = reinterpret_tensor(buf359, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf359  # reuse
        # Source Nodes: [sp_59], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf401, buf402, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf403 = buf363; del buf363  # reuse
        buf404 = buf362; del buf362  # reuse
        buf405 = buf361; del buf361  # reuse
        # Source Nodes: [sp_60], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf402, buf403, buf404, buf405, 2744, 128, grid=grid(2744), stream=stream0)
        buf406 = buf366; del buf366  # reuse
        buf407 = buf365; del buf365  # reuse
        buf408 = buf364; del buf364  # reuse
        # Source Nodes: [sp_60], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf403, buf404, buf405, buf406, buf407, buf408, 28, 98, grid=grid(28), stream=stream0)
        buf409 = buf368; del buf368  # reuse
        buf410 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf412 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_60], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf406, buf407, buf408, primals_513, primals_514, buf409, buf410, buf412, primals_513, primals_514, 14, 2, grid=grid(14), stream=stream0)
        del primals_513
        del primals_514
        buf499 = empty((8, 112, 56, 56), device='cuda', dtype=torch.float32)
        buf413 = reinterpret_tensor(buf499, (8, 14, 56, 56), (351232, 3136, 56, 1), 0)  # alias
        buf1930 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_60, sp_61], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf402, buf409, buf410, primals_65, primals_66, buf413, buf1930, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_66
        buf414 = reinterpret_tensor(buf401, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf401  # reuse
        # Source Nodes: [sp_62], Original ATen: [aten.add]
        triton_poi_fused_add_28.run(buf413, buf400, buf414, 25088, 14, grid=grid(25088, 14), stream=stream0)
        # Source Nodes: [sp_63], Original ATen: [aten.convolution]
        buf415 = extern_kernels.convolution(buf414, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf415, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf416 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_63], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf415, buf416, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf417 = buf405; del buf405  # reuse
        buf418 = buf404; del buf404  # reuse
        buf419 = buf403; del buf403  # reuse
        # Source Nodes: [sp_64], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf416, buf417, buf418, buf419, 2744, 128, grid=grid(2744), stream=stream0)
        buf420 = buf408; del buf408  # reuse
        buf421 = buf407; del buf407  # reuse
        buf422 = buf406; del buf406  # reuse
        # Source Nodes: [sp_64], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf417, buf418, buf419, buf420, buf421, buf422, 28, 98, grid=grid(28), stream=stream0)
        buf423 = buf410; del buf410  # reuse
        buf424 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf426 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_64], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf420, buf421, buf422, primals_516, primals_517, buf423, buf424, buf426, primals_516, primals_517, 14, 2, grid=grid(14), stream=stream0)
        del primals_516
        del primals_517
        buf427 = reinterpret_tensor(buf499, (8, 14, 56, 56), (351232, 3136, 56, 1), 43904)  # alias
        buf1929 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_64, sp_65], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf416, buf423, buf424, primals_68, primals_69, buf427, buf1929, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_69
        buf428 = reinterpret_tensor(buf415, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf415  # reuse
        # Source Nodes: [sp_66], Original ATen: [aten.add]
        triton_poi_fused_add_29.run(buf427, buf400, buf428, 25088, 14, grid=grid(25088, 14), stream=stream0)
        # Source Nodes: [sp_67], Original ATen: [aten.convolution]
        buf429 = extern_kernels.convolution(buf428, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf429, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf430 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_67], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf429, buf430, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf431 = buf419; del buf419  # reuse
        buf432 = buf418; del buf418  # reuse
        buf433 = buf417; del buf417  # reuse
        # Source Nodes: [sp_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf430, buf431, buf432, buf433, 2744, 128, grid=grid(2744), stream=stream0)
        buf434 = buf422; del buf422  # reuse
        buf435 = buf421; del buf421  # reuse
        buf436 = buf420; del buf420  # reuse
        # Source Nodes: [sp_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf431, buf432, buf433, buf434, buf435, buf436, 28, 98, grid=grid(28), stream=stream0)
        buf437 = buf424; del buf424  # reuse
        buf438 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf440 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf434, buf435, buf436, primals_519, primals_520, buf437, buf438, buf440, primals_519, primals_520, 14, 2, grid=grid(14), stream=stream0)
        del primals_519
        del primals_520
        buf441 = reinterpret_tensor(buf499, (8, 14, 56, 56), (351232, 3136, 56, 1), 87808)  # alias
        buf1928 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_68, sp_69], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf430, buf437, buf438, primals_71, primals_72, buf441, buf1928, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_72
        buf442 = reinterpret_tensor(buf429, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf429  # reuse
        # Source Nodes: [sp_70], Original ATen: [aten.add]
        triton_poi_fused_add_30.run(buf441, buf400, buf442, 25088, 14, grid=grid(25088, 14), stream=stream0)
        # Source Nodes: [sp_71], Original ATen: [aten.convolution]
        buf443 = extern_kernels.convolution(buf442, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf443, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf444 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_71], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf443, buf444, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf445 = buf433; del buf433  # reuse
        buf446 = buf432; del buf432  # reuse
        buf447 = buf431; del buf431  # reuse
        # Source Nodes: [sp_72], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf444, buf445, buf446, buf447, 2744, 128, grid=grid(2744), stream=stream0)
        buf448 = buf436; del buf436  # reuse
        buf449 = buf435; del buf435  # reuse
        buf450 = buf434; del buf434  # reuse
        # Source Nodes: [sp_72], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf445, buf446, buf447, buf448, buf449, buf450, 28, 98, grid=grid(28), stream=stream0)
        buf451 = buf438; del buf438  # reuse
        buf452 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf454 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_72], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf448, buf449, buf450, primals_522, primals_523, buf451, buf452, buf454, primals_522, primals_523, 14, 2, grid=grid(14), stream=stream0)
        del primals_522
        del primals_523
        buf455 = reinterpret_tensor(buf499, (8, 14, 56, 56), (351232, 3136, 56, 1), 131712)  # alias
        buf1927 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_72, sp_73], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf444, buf451, buf452, primals_74, primals_75, buf455, buf1927, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_75
        buf456 = reinterpret_tensor(buf443, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf443  # reuse
        # Source Nodes: [sp_74], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(buf455, buf400, buf456, 25088, 14, grid=grid(25088, 14), stream=stream0)
        # Source Nodes: [sp_75], Original ATen: [aten.convolution]
        buf457 = extern_kernels.convolution(buf456, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf457, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf458 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_75], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf457, buf458, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf459 = buf447; del buf447  # reuse
        buf460 = buf446; del buf446  # reuse
        buf461 = buf445; del buf445  # reuse
        # Source Nodes: [sp_76], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf458, buf459, buf460, buf461, 2744, 128, grid=grid(2744), stream=stream0)
        buf462 = buf450; del buf450  # reuse
        buf463 = buf449; del buf449  # reuse
        buf464 = buf448; del buf448  # reuse
        # Source Nodes: [sp_76], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf459, buf460, buf461, buf462, buf463, buf464, 28, 98, grid=grid(28), stream=stream0)
        buf465 = buf452; del buf452  # reuse
        buf466 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf468 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_76], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf462, buf463, buf464, primals_525, primals_526, buf465, buf466, buf468, primals_525, primals_526, 14, 2, grid=grid(14), stream=stream0)
        del primals_525
        del primals_526
        buf469 = reinterpret_tensor(buf499, (8, 14, 56, 56), (351232, 3136, 56, 1), 175616)  # alias
        buf1926 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_76, sp_77], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf458, buf465, buf466, primals_77, primals_78, buf469, buf1926, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_78
        buf470 = reinterpret_tensor(buf457, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf457  # reuse
        # Source Nodes: [sp_78], Original ATen: [aten.add]
        triton_poi_fused_add_32.run(buf469, buf400, buf470, 25088, 14, grid=grid(25088, 14), stream=stream0)
        # Source Nodes: [sp_79], Original ATen: [aten.convolution]
        buf471 = extern_kernels.convolution(buf470, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf471, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf472 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_79], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf471, buf472, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf473 = buf461; del buf461  # reuse
        buf474 = buf460; del buf460  # reuse
        buf475 = buf459; del buf459  # reuse
        # Source Nodes: [sp_80], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf472, buf473, buf474, buf475, 2744, 128, grid=grid(2744), stream=stream0)
        buf476 = buf464; del buf464  # reuse
        buf477 = buf463; del buf463  # reuse
        buf478 = buf462; del buf462  # reuse
        # Source Nodes: [sp_80], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf473, buf474, buf475, buf476, buf477, buf478, 28, 98, grid=grid(28), stream=stream0)
        buf479 = buf466; del buf466  # reuse
        buf480 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf482 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_80], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf476, buf477, buf478, primals_528, primals_529, buf479, buf480, buf482, primals_528, primals_529, 14, 2, grid=grid(14), stream=stream0)
        del primals_528
        del primals_529
        buf483 = reinterpret_tensor(buf499, (8, 14, 56, 56), (351232, 3136, 56, 1), 219520)  # alias
        buf1925 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_80, sp_81], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf472, buf479, buf480, primals_80, primals_81, buf483, buf1925, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del primals_81
        buf484 = reinterpret_tensor(buf471, (8, 14, 56, 56), (43904, 1, 784, 14), 0); del buf471  # reuse
        # Source Nodes: [sp_82], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf483, buf400, buf484, 25088, 14, grid=grid(25088, 14), stream=stream0)
        # Source Nodes: [sp_83], Original ATen: [aten.convolution]
        buf485 = extern_kernels.convolution(buf484, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf485, (8, 14, 56, 56), (43904, 3136, 56, 1))
        buf486 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_83], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf485, buf486, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf487 = buf475; del buf475  # reuse
        buf488 = buf474; del buf474  # reuse
        buf489 = buf473; del buf473  # reuse
        # Source Nodes: [sp_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf486, buf487, buf488, buf489, 2744, 128, grid=grid(2744), stream=stream0)
        buf490 = buf478; del buf478  # reuse
        buf491 = buf477; del buf477  # reuse
        buf492 = buf476; del buf476  # reuse
        # Source Nodes: [sp_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf487, buf488, buf489, buf490, buf491, buf492, 28, 98, grid=grid(28), stream=stream0)
        del buf487
        del buf488
        del buf489
        buf493 = buf480; del buf480  # reuse
        buf494 = empty_strided((1, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        buf496 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf490, buf491, buf492, primals_531, primals_532, buf493, buf494, buf496, primals_531, primals_532, 14, 2, grid=grid(14), stream=stream0)
        del primals_531
        del primals_532
        buf497 = reinterpret_tensor(buf499, (8, 14, 56, 56), (351232, 3136, 56, 1), 263424)  # alias
        buf1924 = empty_strided((8, 14, 56, 56), (43904, 1, 784, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_84, sp_85], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_21.run(buf486, buf493, buf494, primals_83, primals_84, buf497, buf1924, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del buf494
        del primals_84
        buf498 = reinterpret_tensor(buf499, (8, 14, 56, 56), (351232, 3136, 56, 1), 307328)  # alias
        # Source Nodes: [cat_29], Original ATen: [aten.cat]
        triton_poi_fused_cat_34.run(buf400, buf498, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf500 = empty_strided((8, 112, 56, 56), (351232, 1, 6272, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_29], Original ATen: [aten.cat]
        triton_poi_fused_convolution_12.run(buf499, buf500, 896, 3136, grid=grid(896, 3136), stream=stream0)
        del buf413
        del buf427
        del buf441
        del buf455
        del buf469
        del buf483
        del buf497
        del buf498
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        buf501 = extern_kernels.convolution(buf500, primals_85, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf501, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf502 = empty_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf501, buf502, 2048, 3136, grid=grid(2048, 3136), stream=stream0)
        buf503 = buf379; del buf379  # reuse
        buf504 = buf378; del buf378  # reuse
        buf505 = buf377; del buf377  # reuse
        # Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf502, buf503, buf504, buf505, 50176, 128, grid=grid(50176), stream=stream0)
        buf506 = buf382; del buf382  # reuse
        buf507 = buf381; del buf381  # reuse
        buf508 = buf380; del buf380  # reuse
        # Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf503, buf504, buf505, buf506, buf507, buf508, 512, 98, grid=grid(512), stream=stream0)
        del buf503
        del buf504
        del buf505
        buf509 = buf384; del buf384  # reuse
        buf510 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf512 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf506, buf507, buf508, primals_534, primals_535, buf509, buf510, buf512, primals_534, primals_535, 256, 2, grid=grid(256), stream=stream0)
        del primals_534
        del primals_535
        buf513 = reinterpret_tensor(buf501, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf501  # reuse
        # Source Nodes: [out_21, out_22, shortcut_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_35.run(buf502, buf509, buf510, primals_86, primals_87, buf387, buf513, 6422528, grid=grid(6422528), stream=stream0)
        del buf510
        del primals_87
        # Source Nodes: [out_24], Original ATen: [aten.convolution]
        buf514 = extern_kernels.convolution(buf513, primals_88, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf514, (8, 224, 56, 56), (702464, 3136, 56, 1))
        buf515 = empty_strided((8, 224, 56, 56), (702464, 1, 12544, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_24], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf514, buf515, 1792, 3136, grid=grid(1792, 3136), stream=stream0)
        buf516 = empty_strided((1, 224, 1, 1, 196), (43904, 1, 43904, 43904, 224), device='cuda', dtype=torch.float32)
        buf517 = empty_strided((1, 224, 1, 1, 196), (43904, 1, 43904, 43904, 224), device='cuda', dtype=torch.float32)
        buf518 = empty_strided((1, 224, 1, 1, 196), (43904, 1, 43904, 43904, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf515, buf516, buf517, buf518, 43904, 128, grid=grid(43904), stream=stream0)
        buf519 = reinterpret_tensor(buf121, (1, 224, 1, 1, 2), (448, 1, 448, 448, 224), 0); del buf121  # reuse
        buf520 = reinterpret_tensor(buf120, (1, 224, 1, 1, 2), (448, 1, 448, 448, 224), 0); del buf120  # reuse
        buf521 = reinterpret_tensor(buf119, (1, 224, 1, 1, 2), (448, 1, 448, 448, 224), 0); del buf119  # reuse
        # Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf516, buf517, buf518, buf519, buf520, buf521, 448, 98, grid=grid(448), stream=stream0)
        buf522 = reinterpret_tensor(buf395, (1, 224, 1, 1), (224, 1, 224, 224), 0); del buf395  # reuse
        buf523 = reinterpret_tensor(buf394, (1, 224, 1, 1), (224, 1, 224, 224), 0); del buf394  # reuse
        buf525 = reinterpret_tensor(buf393, (224, ), (1, ), 0); del buf393  # reuse
        # Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf519, buf520, buf521, primals_537, primals_538, buf522, buf523, buf525, primals_537, primals_538, 224, 2, grid=grid(224), stream=stream0)
        del primals_537
        del primals_538
        buf526 = reinterpret_tensor(buf514, (8, 224, 56, 56), (702464, 1, 12544, 224), 0); del buf514  # reuse
        buf1923 = empty_strided((8, 224, 56, 56), (702464, 1, 12544, 224), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_25, out_26], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_40.run(buf515, buf522, buf523, primals_89, primals_90, buf526, buf1923, 5619712, grid=grid(5619712), stream=stream0)
        del primals_90
        # Source Nodes: [sp_88], Original ATen: [aten.convolution]
        buf527 = extern_kernels.convolution(reinterpret_tensor(buf526, (8, 28, 56, 56), (702464, 1, 12544, 224), 0), buf22, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf527, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf528 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_88], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf527, buf528, 224, 784, grid=grid(224, 784), stream=stream0)
        buf529 = empty_strided((1, 28, 1, 1, 49), (1372, 1, 1372, 1372, 28), device='cuda', dtype=torch.float32)
        buf530 = empty_strided((1, 28, 1, 1, 49), (1372, 1, 1372, 1372, 28), device='cuda', dtype=torch.float32)
        buf531 = empty_strided((1, 28, 1, 1, 49), (1372, 1, 1372, 1372, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf528, buf529, buf530, buf531, 1372, 128, grid=grid(1372), stream=stream0)
        buf532 = reinterpret_tensor(buf492, (1, 28, 1, 1), (28, 1, 28, 28), 0); del buf492  # reuse
        buf533 = reinterpret_tensor(buf491, (1, 28, 1, 1), (28, 1, 28, 28), 0); del buf491  # reuse
        buf535 = reinterpret_tensor(buf490, (28, ), (1, ), 0); del buf490  # reuse
        # Source Nodes: [sp_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf529, buf530, buf531, primals_540, primals_541, buf532, buf533, buf535, primals_540, primals_541, 28, 49, grid=grid(28), stream=stream0)
        del primals_540
        del primals_541
        buf598 = empty((8, 224, 28, 28), device='cuda', dtype=torch.float32)
        buf536 = reinterpret_tensor(buf598, (8, 28, 28, 28), (175616, 784, 28, 1), 0)  # alias
        buf1922 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_89, sp_90], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf528, buf532, buf533, primals_92, primals_93, buf536, buf1922, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_93
        # Source Nodes: [sp_92], Original ATen: [aten.convolution]
        buf537 = extern_kernels.convolution(reinterpret_tensor(buf526, (8, 28, 56, 56), (702464, 1, 12544, 224), 28), buf23, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf537, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf538 = reinterpret_tensor(buf527, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf527  # reuse
        # Source Nodes: [sp_92], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf537, buf538, 224, 784, grid=grid(224, 784), stream=stream0)
        buf539 = buf531; del buf531  # reuse
        buf540 = buf530; del buf530  # reuse
        buf541 = buf529; del buf529  # reuse
        # Source Nodes: [sp_93], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf538, buf539, buf540, buf541, 1372, 128, grid=grid(1372), stream=stream0)
        buf542 = buf533; del buf533  # reuse
        buf543 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf545 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_93], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf539, buf540, buf541, primals_543, primals_544, buf542, buf543, buf545, primals_543, primals_544, 28, 49, grid=grid(28), stream=stream0)
        del primals_543
        del primals_544
        buf546 = reinterpret_tensor(buf598, (8, 28, 28, 28), (175616, 784, 28, 1), 21952)  # alias
        buf1921 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_93, sp_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf538, buf542, buf543, primals_95, primals_96, buf546, buf1921, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_96
        # Source Nodes: [sp_96], Original ATen: [aten.convolution]
        buf547 = extern_kernels.convolution(reinterpret_tensor(buf526, (8, 28, 56, 56), (702464, 1, 12544, 224), 56), buf24, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf547, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf548 = reinterpret_tensor(buf537, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf537  # reuse
        # Source Nodes: [sp_96], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf547, buf548, 224, 784, grid=grid(224, 784), stream=stream0)
        buf549 = buf541; del buf541  # reuse
        buf550 = buf540; del buf540  # reuse
        buf551 = buf539; del buf539  # reuse
        # Source Nodes: [sp_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf548, buf549, buf550, buf551, 1372, 128, grid=grid(1372), stream=stream0)
        buf552 = buf543; del buf543  # reuse
        buf553 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf555 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf549, buf550, buf551, primals_546, primals_547, buf552, buf553, buf555, primals_546, primals_547, 28, 49, grid=grid(28), stream=stream0)
        del primals_546
        del primals_547
        buf556 = reinterpret_tensor(buf598, (8, 28, 28, 28), (175616, 784, 28, 1), 43904)  # alias
        buf1920 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_97, sp_98], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf548, buf552, buf553, primals_98, primals_99, buf556, buf1920, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_99
        # Source Nodes: [sp_100], Original ATen: [aten.convolution]
        buf557 = extern_kernels.convolution(reinterpret_tensor(buf526, (8, 28, 56, 56), (702464, 1, 12544, 224), 84), buf25, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf557, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf558 = reinterpret_tensor(buf547, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf547  # reuse
        # Source Nodes: [sp_100], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf557, buf558, 224, 784, grid=grid(224, 784), stream=stream0)
        buf559 = buf551; del buf551  # reuse
        buf560 = buf550; del buf550  # reuse
        buf561 = buf549; del buf549  # reuse
        # Source Nodes: [sp_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf558, buf559, buf560, buf561, 1372, 128, grid=grid(1372), stream=stream0)
        buf562 = buf553; del buf553  # reuse
        buf563 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf565 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf559, buf560, buf561, primals_549, primals_550, buf562, buf563, buf565, primals_549, primals_550, 28, 49, grid=grid(28), stream=stream0)
        del primals_549
        del primals_550
        buf566 = reinterpret_tensor(buf598, (8, 28, 28, 28), (175616, 784, 28, 1), 65856)  # alias
        buf1919 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_101, sp_102], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf558, buf562, buf563, primals_101, primals_102, buf566, buf1919, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_102
        # Source Nodes: [sp_104], Original ATen: [aten.convolution]
        buf567 = extern_kernels.convolution(reinterpret_tensor(buf526, (8, 28, 56, 56), (702464, 1, 12544, 224), 112), buf26, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf567, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf568 = reinterpret_tensor(buf557, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf557  # reuse
        # Source Nodes: [sp_104], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf567, buf568, 224, 784, grid=grid(224, 784), stream=stream0)
        buf569 = buf561; del buf561  # reuse
        buf570 = buf560; del buf560  # reuse
        buf571 = buf559; del buf559  # reuse
        # Source Nodes: [sp_105], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf568, buf569, buf570, buf571, 1372, 128, grid=grid(1372), stream=stream0)
        buf572 = buf563; del buf563  # reuse
        buf573 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf575 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_105], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf569, buf570, buf571, primals_552, primals_553, buf572, buf573, buf575, primals_552, primals_553, 28, 49, grid=grid(28), stream=stream0)
        del primals_552
        del primals_553
        buf576 = reinterpret_tensor(buf598, (8, 28, 28, 28), (175616, 784, 28, 1), 87808)  # alias
        buf1918 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_105, sp_106], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf568, buf572, buf573, primals_104, primals_105, buf576, buf1918, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_105
        # Source Nodes: [sp_108], Original ATen: [aten.convolution]
        buf577 = extern_kernels.convolution(reinterpret_tensor(buf526, (8, 28, 56, 56), (702464, 1, 12544, 224), 140), buf27, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf577, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf578 = reinterpret_tensor(buf567, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf567  # reuse
        # Source Nodes: [sp_108], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf577, buf578, 224, 784, grid=grid(224, 784), stream=stream0)
        buf579 = buf571; del buf571  # reuse
        buf580 = buf570; del buf570  # reuse
        buf581 = buf569; del buf569  # reuse
        # Source Nodes: [sp_109], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf578, buf579, buf580, buf581, 1372, 128, grid=grid(1372), stream=stream0)
        buf582 = buf573; del buf573  # reuse
        buf583 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf585 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_109], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf579, buf580, buf581, primals_555, primals_556, buf582, buf583, buf585, primals_555, primals_556, 28, 49, grid=grid(28), stream=stream0)
        del primals_555
        del primals_556
        buf586 = reinterpret_tensor(buf598, (8, 28, 28, 28), (175616, 784, 28, 1), 109760)  # alias
        buf1917 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_109, sp_110], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf578, buf582, buf583, primals_107, primals_108, buf586, buf1917, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_108
        # Source Nodes: [sp_112], Original ATen: [aten.convolution]
        buf587 = extern_kernels.convolution(reinterpret_tensor(buf526, (8, 28, 56, 56), (702464, 1, 12544, 224), 168), buf28, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf587, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf588 = reinterpret_tensor(buf577, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf577  # reuse
        # Source Nodes: [sp_112], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf587, buf588, 224, 784, grid=grid(224, 784), stream=stream0)
        buf589 = buf581; del buf581  # reuse
        buf590 = buf580; del buf580  # reuse
        buf591 = buf579; del buf579  # reuse
        # Source Nodes: [sp_113], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf588, buf589, buf590, buf591, 1372, 128, grid=grid(1372), stream=stream0)
        buf592 = buf583; del buf583  # reuse
        buf593 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf595 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_113], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf589, buf590, buf591, primals_558, primals_559, buf592, buf593, buf595, primals_558, primals_559, 28, 49, grid=grid(28), stream=stream0)
        del primals_558
        del primals_559
        buf596 = reinterpret_tensor(buf598, (8, 28, 28, 28), (175616, 784, 28, 1), 131712)  # alias
        buf1916 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_113, sp_114], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf588, buf592, buf593, primals_110, primals_111, buf596, buf1916, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_111
        buf597 = reinterpret_tensor(buf598, (8, 28, 28, 28), (175616, 784, 28, 1), 153664)  # alias
        # Source Nodes: [getattr_l__mod___layer2___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_45.run(buf526, buf597, 224, 784, grid=grid(224, 784), stream=stream0)
        buf599 = empty_strided((8, 224, 28, 28), (175616, 1, 6272, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_28], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf598, buf599, 1792, 784, grid=grid(1792, 784), stream=stream0)
        del buf536
        del buf546
        del buf556
        del buf566
        del buf576
        del buf586
        del buf596
        del buf597
        # Source Nodes: [out_28], Original ATen: [aten.convolution]
        buf600 = extern_kernels.convolution(buf599, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf600, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf601 = empty_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_28], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf600, buf601, 4096, 784, grid=grid(4096, 784), stream=stream0)
        buf602 = empty_strided((1, 512, 1, 1, 49), (25088, 1, 25088, 25088, 512), device='cuda', dtype=torch.float32)
        buf603 = empty_strided((1, 512, 1, 1, 49), (25088, 1, 25088, 25088, 512), device='cuda', dtype=torch.float32)
        buf604 = empty_strided((1, 512, 1, 1, 49), (25088, 1, 25088, 25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_48.run(buf601, buf602, buf603, buf604, 25088, 128, grid=grid(25088), stream=stream0)
        buf605 = reinterpret_tensor(buf508, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf508  # reuse
        buf606 = reinterpret_tensor(buf507, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf507  # reuse
        buf608 = reinterpret_tensor(buf506, (512, ), (1, ), 0); del buf506  # reuse
        # Source Nodes: [out_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_49.run(buf602, buf603, buf604, primals_561, primals_562, buf605, buf606, buf608, primals_561, primals_562, 512, 49, grid=grid(512), stream=stream0)
        del primals_561
        del primals_562
        # Source Nodes: [getattr_l__mod___layer2___0___downsample_0], Original ATen: [aten.convolution]
        buf609 = extern_kernels.convolution(buf513, primals_115, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf609, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf610 = reinterpret_tensor(buf600, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf600  # reuse
        # Source Nodes: [getattr_l__mod___layer2___0___downsample_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf609, buf610, 4096, 784, grid=grid(4096, 784), stream=stream0)
        buf611 = buf604; del buf604  # reuse
        buf612 = buf603; del buf603  # reuse
        buf613 = buf602; del buf602  # reuse
        # Source Nodes: [shortcut_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_48.run(buf610, buf611, buf612, buf613, 25088, 128, grid=grid(25088), stream=stream0)
        buf614 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf615 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf617 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_49.run(buf611, buf612, buf613, primals_564, primals_565, buf614, buf615, buf617, primals_564, primals_565, 512, 49, grid=grid(512), stream=stream0)
        del primals_564
        del primals_565
        buf618 = reinterpret_tensor(buf609, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf609  # reuse
        buf619 = buf618; del buf618  # reuse
        # Source Nodes: [out_29, out_30, shortcut_5, shortcut_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_50.run(buf619, buf601, buf605, buf606, primals_113, primals_114, buf610, buf614, buf615, primals_116, primals_117, 3211264, grid=grid(3211264), stream=stream0)
        del primals_114
        del primals_117
        # Source Nodes: [out_32], Original ATen: [aten.convolution]
        buf620 = extern_kernels.convolution(buf619, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf620, (8, 224, 28, 28), (175616, 784, 28, 1))
        buf621 = reinterpret_tensor(buf598, (8, 224, 28, 28), (175616, 1, 6272, 224), 0); del buf598  # reuse
        # Source Nodes: [out_32], Original ATen: [aten.convolution]
        triton_poi_fused_cat_46.run(buf620, buf621, 1792, 784, grid=grid(1792, 784), stream=stream0)
        buf622 = empty_strided((1, 224, 1, 1, 49), (10976, 1, 10976, 10976, 224), device='cuda', dtype=torch.float32)
        buf623 = empty_strided((1, 224, 1, 1, 49), (10976, 1, 10976, 10976, 224), device='cuda', dtype=torch.float32)
        buf624 = empty_strided((1, 224, 1, 1, 49), (10976, 1, 10976, 10976, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf621, buf622, buf623, buf624, 10976, 128, grid=grid(10976), stream=stream0)
        buf625 = buf523; del buf523  # reuse
        buf626 = empty_strided((1, 224, 1, 1), (224, 1, 224, 224), device='cuda', dtype=torch.float32)
        buf628 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_52.run(buf622, buf623, buf624, primals_567, primals_568, buf625, buf626, buf628, primals_567, primals_568, 224, 49, grid=grid(224), stream=stream0)
        del primals_567
        del primals_568
        buf629 = reinterpret_tensor(buf620, (8, 224, 28, 28), (175616, 1, 6272, 224), 0); del buf620  # reuse
        buf1915 = empty_strided((8, 224, 28, 28), (175616, 1, 6272, 224), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_33, out_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_53.run(buf621, buf625, buf626, primals_119, primals_120, buf629, buf1915, 1404928, grid=grid(1404928), stream=stream0)
        del primals_120
        # Source Nodes: [sp_117], Original ATen: [aten.convolution]
        buf630 = extern_kernels.convolution(reinterpret_tensor(buf629, (8, 28, 28, 28), (175616, 1, 6272, 224), 0), buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf630, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf631 = reinterpret_tensor(buf587, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf587  # reuse
        # Source Nodes: [sp_117], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf630, buf631, 224, 784, grid=grid(224, 784), stream=stream0)
        buf632 = buf591; del buf591  # reuse
        buf633 = buf590; del buf590  # reuse
        buf634 = buf589; del buf589  # reuse
        # Source Nodes: [sp_118], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf631, buf632, buf633, buf634, 1372, 128, grid=grid(1372), stream=stream0)
        buf635 = buf593; del buf593  # reuse
        buf636 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf638 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_118], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf632, buf633, buf634, primals_570, primals_571, buf635, buf636, buf638, primals_570, primals_571, 28, 49, grid=grid(28), stream=stream0)
        del primals_570
        del primals_571
        buf707 = empty((8, 224, 28, 28), device='cuda', dtype=torch.float32)
        buf639 = reinterpret_tensor(buf707, (8, 28, 28, 28), (175616, 784, 28, 1), 0)  # alias
        buf1914 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_118, sp_119], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf631, buf635, buf636, primals_122, primals_123, buf639, buf1914, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_123
        buf640 = reinterpret_tensor(buf630, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf630  # reuse
        # Source Nodes: [sp_120], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(buf639, buf629, buf640, 6272, 28, grid=grid(6272, 28), stream=stream0)
        # Source Nodes: [sp_121], Original ATen: [aten.convolution]
        buf641 = extern_kernels.convolution(buf640, buf30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf641, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf642 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_121], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf641, buf642, 224, 784, grid=grid(224, 784), stream=stream0)
        buf643 = buf634; del buf634  # reuse
        buf644 = buf633; del buf633  # reuse
        buf645 = buf632; del buf632  # reuse
        # Source Nodes: [sp_122], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf642, buf643, buf644, buf645, 1372, 128, grid=grid(1372), stream=stream0)
        buf646 = buf636; del buf636  # reuse
        buf647 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf649 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_122], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf643, buf644, buf645, primals_573, primals_574, buf646, buf647, buf649, primals_573, primals_574, 28, 49, grid=grid(28), stream=stream0)
        del primals_573
        del primals_574
        buf650 = reinterpret_tensor(buf707, (8, 28, 28, 28), (175616, 784, 28, 1), 21952)  # alias
        buf1913 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_122, sp_123], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf642, buf646, buf647, primals_125, primals_126, buf650, buf1913, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_126
        buf651 = reinterpret_tensor(buf641, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf641  # reuse
        # Source Nodes: [sp_124], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(buf650, buf629, buf651, 6272, 28, grid=grid(6272, 28), stream=stream0)
        # Source Nodes: [sp_125], Original ATen: [aten.convolution]
        buf652 = extern_kernels.convolution(buf651, buf31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf652, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf653 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_125], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf652, buf653, 224, 784, grid=grid(224, 784), stream=stream0)
        buf654 = buf645; del buf645  # reuse
        buf655 = buf644; del buf644  # reuse
        buf656 = buf643; del buf643  # reuse
        # Source Nodes: [sp_126], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf653, buf654, buf655, buf656, 1372, 128, grid=grid(1372), stream=stream0)
        buf657 = buf647; del buf647  # reuse
        buf658 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf660 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_126], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf654, buf655, buf656, primals_576, primals_577, buf657, buf658, buf660, primals_576, primals_577, 28, 49, grid=grid(28), stream=stream0)
        del primals_576
        del primals_577
        buf661 = reinterpret_tensor(buf707, (8, 28, 28, 28), (175616, 784, 28, 1), 43904)  # alias
        buf1912 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_126, sp_127], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf653, buf657, buf658, primals_128, primals_129, buf661, buf1912, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_129
        buf662 = reinterpret_tensor(buf652, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf652  # reuse
        # Source Nodes: [sp_128], Original ATen: [aten.add]
        triton_poi_fused_add_56.run(buf661, buf629, buf662, 6272, 28, grid=grid(6272, 28), stream=stream0)
        # Source Nodes: [sp_129], Original ATen: [aten.convolution]
        buf663 = extern_kernels.convolution(buf662, buf32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf663, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf664 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_129], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf663, buf664, 224, 784, grid=grid(224, 784), stream=stream0)
        buf665 = buf656; del buf656  # reuse
        buf666 = buf655; del buf655  # reuse
        buf667 = buf654; del buf654  # reuse
        # Source Nodes: [sp_130], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf664, buf665, buf666, buf667, 1372, 128, grid=grid(1372), stream=stream0)
        buf668 = buf658; del buf658  # reuse
        buf669 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf671 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_130], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf665, buf666, buf667, primals_579, primals_580, buf668, buf669, buf671, primals_579, primals_580, 28, 49, grid=grid(28), stream=stream0)
        del primals_579
        del primals_580
        buf672 = reinterpret_tensor(buf707, (8, 28, 28, 28), (175616, 784, 28, 1), 65856)  # alias
        buf1911 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_130, sp_131], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf664, buf668, buf669, primals_131, primals_132, buf672, buf1911, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_132
        buf673 = reinterpret_tensor(buf663, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf663  # reuse
        # Source Nodes: [sp_132], Original ATen: [aten.add]
        triton_poi_fused_add_57.run(buf672, buf629, buf673, 6272, 28, grid=grid(6272, 28), stream=stream0)
        # Source Nodes: [sp_133], Original ATen: [aten.convolution]
        buf674 = extern_kernels.convolution(buf673, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf674, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf675 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_133], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf674, buf675, 224, 784, grid=grid(224, 784), stream=stream0)
        buf676 = buf667; del buf667  # reuse
        buf677 = buf666; del buf666  # reuse
        buf678 = buf665; del buf665  # reuse
        # Source Nodes: [sp_134], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf675, buf676, buf677, buf678, 1372, 128, grid=grid(1372), stream=stream0)
        buf679 = buf669; del buf669  # reuse
        buf680 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf682 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_134], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf676, buf677, buf678, primals_582, primals_583, buf679, buf680, buf682, primals_582, primals_583, 28, 49, grid=grid(28), stream=stream0)
        del primals_582
        del primals_583
        buf683 = reinterpret_tensor(buf707, (8, 28, 28, 28), (175616, 784, 28, 1), 87808)  # alias
        buf1910 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_134, sp_135], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf675, buf679, buf680, primals_134, primals_135, buf683, buf1910, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_135
        buf684 = reinterpret_tensor(buf674, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf674  # reuse
        # Source Nodes: [sp_136], Original ATen: [aten.add]
        triton_poi_fused_add_58.run(buf683, buf629, buf684, 6272, 28, grid=grid(6272, 28), stream=stream0)
        # Source Nodes: [sp_137], Original ATen: [aten.convolution]
        buf685 = extern_kernels.convolution(buf684, buf34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf685, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf686 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_137], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf685, buf686, 224, 784, grid=grid(224, 784), stream=stream0)
        buf687 = buf678; del buf678  # reuse
        buf688 = buf677; del buf677  # reuse
        buf689 = buf676; del buf676  # reuse
        # Source Nodes: [sp_138], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf686, buf687, buf688, buf689, 1372, 128, grid=grid(1372), stream=stream0)
        buf690 = buf680; del buf680  # reuse
        buf691 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf693 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_138], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf687, buf688, buf689, primals_585, primals_586, buf690, buf691, buf693, primals_585, primals_586, 28, 49, grid=grid(28), stream=stream0)
        del primals_585
        del primals_586
        buf694 = reinterpret_tensor(buf707, (8, 28, 28, 28), (175616, 784, 28, 1), 109760)  # alias
        buf1909 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_138, sp_139], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf686, buf690, buf691, primals_137, primals_138, buf694, buf1909, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_138
        buf695 = reinterpret_tensor(buf685, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf685  # reuse
        # Source Nodes: [sp_140], Original ATen: [aten.add]
        triton_poi_fused_add_59.run(buf694, buf629, buf695, 6272, 28, grid=grid(6272, 28), stream=stream0)
        # Source Nodes: [sp_141], Original ATen: [aten.convolution]
        buf696 = extern_kernels.convolution(buf695, buf35, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf696, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf697 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_141], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf696, buf697, 224, 784, grid=grid(224, 784), stream=stream0)
        buf698 = buf689; del buf689  # reuse
        buf699 = buf688; del buf688  # reuse
        buf700 = buf687; del buf687  # reuse
        # Source Nodes: [sp_142], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf697, buf698, buf699, buf700, 1372, 128, grid=grid(1372), stream=stream0)
        buf701 = buf691; del buf691  # reuse
        buf702 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf704 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_142], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf698, buf699, buf700, primals_588, primals_589, buf701, buf702, buf704, primals_588, primals_589, 28, 49, grid=grid(28), stream=stream0)
        del primals_588
        del primals_589
        buf705 = reinterpret_tensor(buf707, (8, 28, 28, 28), (175616, 784, 28, 1), 131712)  # alias
        buf1908 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_142, sp_143], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf697, buf701, buf702, primals_140, primals_141, buf705, buf1908, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_141
        buf706 = reinterpret_tensor(buf707, (8, 28, 28, 28), (175616, 784, 28, 1), 153664)  # alias
        # Source Nodes: [cat_27], Original ATen: [aten.cat]
        triton_poi_fused_cat_60.run(buf629, buf706, 224, 784, grid=grid(224, 784), stream=stream0)
        buf708 = empty_strided((8, 224, 28, 28), (175616, 1, 6272, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_27], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf707, buf708, 1792, 784, grid=grid(1792, 784), stream=stream0)
        del buf639
        del buf650
        del buf661
        del buf672
        del buf683
        del buf694
        del buf705
        del buf706
        # Source Nodes: [out_36], Original ATen: [aten.convolution]
        buf709 = extern_kernels.convolution(buf708, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf709, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf710 = empty_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_36], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf709, buf710, 4096, 784, grid=grid(4096, 784), stream=stream0)
        buf711 = buf613; del buf613  # reuse
        buf712 = buf612; del buf612  # reuse
        buf713 = buf611; del buf611  # reuse
        # Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_48.run(buf710, buf711, buf712, buf713, 25088, 128, grid=grid(25088), stream=stream0)
        buf714 = buf615; del buf615  # reuse
        buf715 = buf606; del buf606  # reuse
        buf717 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_49.run(buf711, buf712, buf713, primals_591, primals_592, buf714, buf715, buf717, primals_591, primals_592, 512, 49, grid=grid(512), stream=stream0)
        del primals_591
        del primals_592
        buf718 = reinterpret_tensor(buf709, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf709  # reuse
        # Source Nodes: [out_37, out_38, shortcut_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_61.run(buf710, buf714, buf715, primals_143, primals_144, buf619, buf718, 3211264, grid=grid(3211264), stream=stream0)
        del primals_144
        # Source Nodes: [out_40], Original ATen: [aten.convolution]
        buf719 = extern_kernels.convolution(buf718, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf719, (8, 224, 28, 28), (175616, 784, 28, 1))
        buf720 = reinterpret_tensor(buf707, (8, 224, 28, 28), (175616, 1, 6272, 224), 0); del buf707  # reuse
        # Source Nodes: [out_40], Original ATen: [aten.convolution]
        triton_poi_fused_cat_46.run(buf719, buf720, 1792, 784, grid=grid(1792, 784), stream=stream0)
        buf721 = buf624; del buf624  # reuse
        buf722 = buf623; del buf623  # reuse
        buf723 = buf622; del buf622  # reuse
        # Source Nodes: [out_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf720, buf721, buf722, buf723, 10976, 128, grid=grid(10976), stream=stream0)
        buf724 = buf626; del buf626  # reuse
        buf725 = empty_strided((1, 224, 1, 1), (224, 1, 224, 224), device='cuda', dtype=torch.float32)
        buf727 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_52.run(buf721, buf722, buf723, primals_594, primals_595, buf724, buf725, buf727, primals_594, primals_595, 224, 49, grid=grid(224), stream=stream0)
        del primals_594
        del primals_595
        buf728 = reinterpret_tensor(buf719, (8, 224, 28, 28), (175616, 1, 6272, 224), 0); del buf719  # reuse
        buf1907 = empty_strided((8, 224, 28, 28), (175616, 1, 6272, 224), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_41, out_42], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_53.run(buf720, buf724, buf725, primals_146, primals_147, buf728, buf1907, 1404928, grid=grid(1404928), stream=stream0)
        del primals_147
        # Source Nodes: [sp_146], Original ATen: [aten.convolution]
        buf729 = extern_kernels.convolution(reinterpret_tensor(buf728, (8, 28, 28, 28), (175616, 1, 6272, 224), 0), buf36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf729, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf730 = reinterpret_tensor(buf696, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf696  # reuse
        # Source Nodes: [sp_146], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf729, buf730, 224, 784, grid=grid(224, 784), stream=stream0)
        buf731 = buf700; del buf700  # reuse
        buf732 = buf699; del buf699  # reuse
        buf733 = buf698; del buf698  # reuse
        # Source Nodes: [sp_147], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf730, buf731, buf732, buf733, 1372, 128, grid=grid(1372), stream=stream0)
        buf734 = buf702; del buf702  # reuse
        buf735 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf737 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_147], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf731, buf732, buf733, primals_597, primals_598, buf734, buf735, buf737, primals_597, primals_598, 28, 49, grid=grid(28), stream=stream0)
        del primals_597
        del primals_598
        buf806 = empty((8, 224, 28, 28), device='cuda', dtype=torch.float32)
        buf738 = reinterpret_tensor(buf806, (8, 28, 28, 28), (175616, 784, 28, 1), 0)  # alias
        buf1906 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_147, sp_148], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf730, buf734, buf735, primals_149, primals_150, buf738, buf1906, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_150
        buf739 = reinterpret_tensor(buf729, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf729  # reuse
        # Source Nodes: [sp_149], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(buf738, buf728, buf739, 6272, 28, grid=grid(6272, 28), stream=stream0)
        # Source Nodes: [sp_150], Original ATen: [aten.convolution]
        buf740 = extern_kernels.convolution(buf739, buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf740, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf741 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_150], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf740, buf741, 224, 784, grid=grid(224, 784), stream=stream0)
        buf742 = buf733; del buf733  # reuse
        buf743 = buf732; del buf732  # reuse
        buf744 = buf731; del buf731  # reuse
        # Source Nodes: [sp_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf741, buf742, buf743, buf744, 1372, 128, grid=grid(1372), stream=stream0)
        buf745 = buf735; del buf735  # reuse
        buf746 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf748 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf742, buf743, buf744, primals_600, primals_601, buf745, buf746, buf748, primals_600, primals_601, 28, 49, grid=grid(28), stream=stream0)
        del primals_600
        del primals_601
        buf749 = reinterpret_tensor(buf806, (8, 28, 28, 28), (175616, 784, 28, 1), 21952)  # alias
        buf1905 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_151, sp_152], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf741, buf745, buf746, primals_152, primals_153, buf749, buf1905, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_153
        buf750 = reinterpret_tensor(buf740, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf740  # reuse
        # Source Nodes: [sp_153], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(buf749, buf728, buf750, 6272, 28, grid=grid(6272, 28), stream=stream0)
        # Source Nodes: [sp_154], Original ATen: [aten.convolution]
        buf751 = extern_kernels.convolution(buf750, buf38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf751, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf752 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_154], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf751, buf752, 224, 784, grid=grid(224, 784), stream=stream0)
        buf753 = buf744; del buf744  # reuse
        buf754 = buf743; del buf743  # reuse
        buf755 = buf742; del buf742  # reuse
        # Source Nodes: [sp_155], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf752, buf753, buf754, buf755, 1372, 128, grid=grid(1372), stream=stream0)
        buf756 = buf746; del buf746  # reuse
        buf757 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf759 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_155], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf753, buf754, buf755, primals_603, primals_604, buf756, buf757, buf759, primals_603, primals_604, 28, 49, grid=grid(28), stream=stream0)
        del primals_603
        del primals_604
        buf760 = reinterpret_tensor(buf806, (8, 28, 28, 28), (175616, 784, 28, 1), 43904)  # alias
        buf1904 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_155, sp_156], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf752, buf756, buf757, primals_155, primals_156, buf760, buf1904, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_156
        buf761 = reinterpret_tensor(buf751, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf751  # reuse
        # Source Nodes: [sp_157], Original ATen: [aten.add]
        triton_poi_fused_add_56.run(buf760, buf728, buf761, 6272, 28, grid=grid(6272, 28), stream=stream0)
        # Source Nodes: [sp_158], Original ATen: [aten.convolution]
        buf762 = extern_kernels.convolution(buf761, buf39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf762, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf763 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_158], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf762, buf763, 224, 784, grid=grid(224, 784), stream=stream0)
        buf764 = buf755; del buf755  # reuse
        buf765 = buf754; del buf754  # reuse
        buf766 = buf753; del buf753  # reuse
        # Source Nodes: [sp_159], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf763, buf764, buf765, buf766, 1372, 128, grid=grid(1372), stream=stream0)
        buf767 = buf757; del buf757  # reuse
        buf768 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf770 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_159], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf764, buf765, buf766, primals_606, primals_607, buf767, buf768, buf770, primals_606, primals_607, 28, 49, grid=grid(28), stream=stream0)
        del primals_606
        del primals_607
        buf771 = reinterpret_tensor(buf806, (8, 28, 28, 28), (175616, 784, 28, 1), 65856)  # alias
        buf1903 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_159, sp_160], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf763, buf767, buf768, primals_158, primals_159, buf771, buf1903, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_159
        buf772 = reinterpret_tensor(buf762, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf762  # reuse
        # Source Nodes: [sp_161], Original ATen: [aten.add]
        triton_poi_fused_add_57.run(buf771, buf728, buf772, 6272, 28, grid=grid(6272, 28), stream=stream0)
        # Source Nodes: [sp_162], Original ATen: [aten.convolution]
        buf773 = extern_kernels.convolution(buf772, buf40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf773, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf774 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_162], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf773, buf774, 224, 784, grid=grid(224, 784), stream=stream0)
        buf775 = buf766; del buf766  # reuse
        buf776 = buf765; del buf765  # reuse
        buf777 = buf764; del buf764  # reuse
        # Source Nodes: [sp_163], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf774, buf775, buf776, buf777, 1372, 128, grid=grid(1372), stream=stream0)
        buf778 = buf768; del buf768  # reuse
        buf779 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf781 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_163], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf775, buf776, buf777, primals_609, primals_610, buf778, buf779, buf781, primals_609, primals_610, 28, 49, grid=grid(28), stream=stream0)
        del primals_609
        del primals_610
        buf782 = reinterpret_tensor(buf806, (8, 28, 28, 28), (175616, 784, 28, 1), 87808)  # alias
        buf1902 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_163, sp_164], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf774, buf778, buf779, primals_161, primals_162, buf782, buf1902, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_162
        buf783 = reinterpret_tensor(buf773, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf773  # reuse
        # Source Nodes: [sp_165], Original ATen: [aten.add]
        triton_poi_fused_add_58.run(buf782, buf728, buf783, 6272, 28, grid=grid(6272, 28), stream=stream0)
        # Source Nodes: [sp_166], Original ATen: [aten.convolution]
        buf784 = extern_kernels.convolution(buf783, buf41, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf784, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf785 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_166], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf784, buf785, 224, 784, grid=grid(224, 784), stream=stream0)
        buf786 = buf777; del buf777  # reuse
        buf787 = buf776; del buf776  # reuse
        buf788 = buf775; del buf775  # reuse
        # Source Nodes: [sp_167], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf785, buf786, buf787, buf788, 1372, 128, grid=grid(1372), stream=stream0)
        buf789 = buf779; del buf779  # reuse
        buf790 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf792 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_167], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf786, buf787, buf788, primals_612, primals_613, buf789, buf790, buf792, primals_612, primals_613, 28, 49, grid=grid(28), stream=stream0)
        del primals_612
        del primals_613
        buf793 = reinterpret_tensor(buf806, (8, 28, 28, 28), (175616, 784, 28, 1), 109760)  # alias
        buf1901 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_167, sp_168], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf785, buf789, buf790, primals_164, primals_165, buf793, buf1901, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_165
        buf794 = reinterpret_tensor(buf784, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf784  # reuse
        # Source Nodes: [sp_169], Original ATen: [aten.add]
        triton_poi_fused_add_59.run(buf793, buf728, buf794, 6272, 28, grid=grid(6272, 28), stream=stream0)
        # Source Nodes: [sp_170], Original ATen: [aten.convolution]
        buf795 = extern_kernels.convolution(buf794, buf42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf795, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf796 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_170], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf795, buf796, 224, 784, grid=grid(224, 784), stream=stream0)
        buf797 = buf788; del buf788  # reuse
        buf798 = buf787; del buf787  # reuse
        buf799 = buf786; del buf786  # reuse
        # Source Nodes: [sp_171], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf796, buf797, buf798, buf799, 1372, 128, grid=grid(1372), stream=stream0)
        buf800 = buf790; del buf790  # reuse
        buf801 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf803 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_171], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf797, buf798, buf799, primals_615, primals_616, buf800, buf801, buf803, primals_615, primals_616, 28, 49, grid=grid(28), stream=stream0)
        del primals_615
        del primals_616
        buf804 = reinterpret_tensor(buf806, (8, 28, 28, 28), (175616, 784, 28, 1), 131712)  # alias
        buf1900 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_171, sp_172], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf796, buf800, buf801, primals_167, primals_168, buf804, buf1900, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_168
        buf805 = reinterpret_tensor(buf806, (8, 28, 28, 28), (175616, 784, 28, 1), 153664)  # alias
        # Source Nodes: [cat_26], Original ATen: [aten.cat]
        triton_poi_fused_cat_60.run(buf728, buf805, 224, 784, grid=grid(224, 784), stream=stream0)
        buf807 = empty_strided((8, 224, 28, 28), (175616, 1, 6272, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_26], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf806, buf807, 1792, 784, grid=grid(1792, 784), stream=stream0)
        del buf738
        del buf749
        del buf760
        del buf771
        del buf782
        del buf793
        del buf804
        del buf805
        # Source Nodes: [out_44], Original ATen: [aten.convolution]
        buf808 = extern_kernels.convolution(buf807, primals_169, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf808, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf809 = empty_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_44], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf808, buf809, 4096, 784, grid=grid(4096, 784), stream=stream0)
        buf810 = buf713; del buf713  # reuse
        buf811 = buf712; del buf712  # reuse
        buf812 = buf711; del buf711  # reuse
        # Source Nodes: [out_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_48.run(buf809, buf810, buf811, buf812, 25088, 128, grid=grid(25088), stream=stream0)
        buf813 = buf715; del buf715  # reuse
        buf814 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf816 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_49.run(buf810, buf811, buf812, primals_618, primals_619, buf813, buf814, buf816, primals_618, primals_619, 512, 49, grid=grid(512), stream=stream0)
        del primals_618
        del primals_619
        buf817 = reinterpret_tensor(buf808, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf808  # reuse
        # Source Nodes: [out_45, out_46, shortcut_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_61.run(buf809, buf813, buf814, primals_170, primals_171, buf718, buf817, 3211264, grid=grid(3211264), stream=stream0)
        del primals_171
        # Source Nodes: [out_48], Original ATen: [aten.convolution]
        buf818 = extern_kernels.convolution(buf817, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf818, (8, 224, 28, 28), (175616, 784, 28, 1))
        buf819 = reinterpret_tensor(buf806, (8, 224, 28, 28), (175616, 1, 6272, 224), 0); del buf806  # reuse
        # Source Nodes: [out_48], Original ATen: [aten.convolution]
        triton_poi_fused_cat_46.run(buf818, buf819, 1792, 784, grid=grid(1792, 784), stream=stream0)
        buf820 = buf723; del buf723  # reuse
        buf821 = buf722; del buf722  # reuse
        buf822 = buf721; del buf721  # reuse
        # Source Nodes: [out_49], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf819, buf820, buf821, buf822, 10976, 128, grid=grid(10976), stream=stream0)
        buf823 = buf725; del buf725  # reuse
        buf824 = empty_strided((1, 224, 1, 1), (224, 1, 224, 224), device='cuda', dtype=torch.float32)
        buf826 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_49], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_52.run(buf820, buf821, buf822, primals_621, primals_622, buf823, buf824, buf826, primals_621, primals_622, 224, 49, grid=grid(224), stream=stream0)
        del buf820
        del buf821
        del buf822
        del primals_621
        del primals_622
        buf827 = reinterpret_tensor(buf818, (8, 224, 28, 28), (175616, 1, 6272, 224), 0); del buf818  # reuse
        buf1899 = empty_strided((8, 224, 28, 28), (175616, 1, 6272, 224), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_49, out_50], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_53.run(buf819, buf823, buf824, primals_173, primals_174, buf827, buf1899, 1404928, grid=grid(1404928), stream=stream0)
        del buf824
        del primals_174
        # Source Nodes: [sp_175], Original ATen: [aten.convolution]
        buf828 = extern_kernels.convolution(reinterpret_tensor(buf827, (8, 28, 28, 28), (175616, 1, 6272, 224), 0), buf43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf828, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf829 = reinterpret_tensor(buf795, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf795  # reuse
        # Source Nodes: [sp_175], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf828, buf829, 224, 784, grid=grid(224, 784), stream=stream0)
        buf830 = buf799; del buf799  # reuse
        buf831 = buf798; del buf798  # reuse
        buf832 = buf797; del buf797  # reuse
        # Source Nodes: [sp_176], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf829, buf830, buf831, buf832, 1372, 128, grid=grid(1372), stream=stream0)
        buf833 = buf801; del buf801  # reuse
        buf834 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf836 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_176], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf830, buf831, buf832, primals_624, primals_625, buf833, buf834, buf836, primals_624, primals_625, 28, 49, grid=grid(28), stream=stream0)
        del primals_624
        del primals_625
        buf905 = empty((8, 224, 28, 28), device='cuda', dtype=torch.float32)
        buf837 = reinterpret_tensor(buf905, (8, 28, 28, 28), (175616, 784, 28, 1), 0)  # alias
        buf1898 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_176, sp_177], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf829, buf833, buf834, primals_176, primals_177, buf837, buf1898, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_177
        buf838 = reinterpret_tensor(buf828, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf828  # reuse
        # Source Nodes: [sp_178], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(buf837, buf827, buf838, 6272, 28, grid=grid(6272, 28), stream=stream0)
        # Source Nodes: [sp_179], Original ATen: [aten.convolution]
        buf839 = extern_kernels.convolution(buf838, buf44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf839, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf840 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_179], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf839, buf840, 224, 784, grid=grid(224, 784), stream=stream0)
        buf841 = buf832; del buf832  # reuse
        buf842 = buf831; del buf831  # reuse
        buf843 = buf830; del buf830  # reuse
        # Source Nodes: [sp_180], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf840, buf841, buf842, buf843, 1372, 128, grid=grid(1372), stream=stream0)
        buf844 = buf834; del buf834  # reuse
        buf845 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf847 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_180], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf841, buf842, buf843, primals_627, primals_628, buf844, buf845, buf847, primals_627, primals_628, 28, 49, grid=grid(28), stream=stream0)
        del primals_627
        del primals_628
        buf848 = reinterpret_tensor(buf905, (8, 28, 28, 28), (175616, 784, 28, 1), 21952)  # alias
        buf1897 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_180, sp_181], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf840, buf844, buf845, primals_179, primals_180, buf848, buf1897, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_180
        buf849 = reinterpret_tensor(buf839, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf839  # reuse
        # Source Nodes: [sp_182], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(buf848, buf827, buf849, 6272, 28, grid=grid(6272, 28), stream=stream0)
        # Source Nodes: [sp_183], Original ATen: [aten.convolution]
        buf850 = extern_kernels.convolution(buf849, buf45, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf850, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf851 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_183], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf850, buf851, 224, 784, grid=grid(224, 784), stream=stream0)
        buf852 = buf843; del buf843  # reuse
        buf853 = buf842; del buf842  # reuse
        buf854 = buf841; del buf841  # reuse
        # Source Nodes: [sp_184], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf851, buf852, buf853, buf854, 1372, 128, grid=grid(1372), stream=stream0)
        buf855 = buf845; del buf845  # reuse
        buf856 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf858 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_184], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf852, buf853, buf854, primals_630, primals_631, buf855, buf856, buf858, primals_630, primals_631, 28, 49, grid=grid(28), stream=stream0)
        del primals_630
        del primals_631
        buf859 = reinterpret_tensor(buf905, (8, 28, 28, 28), (175616, 784, 28, 1), 43904)  # alias
        buf1896 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_184, sp_185], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf851, buf855, buf856, primals_182, primals_183, buf859, buf1896, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_183
        buf860 = reinterpret_tensor(buf850, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf850  # reuse
        # Source Nodes: [sp_186], Original ATen: [aten.add]
        triton_poi_fused_add_56.run(buf859, buf827, buf860, 6272, 28, grid=grid(6272, 28), stream=stream0)
        # Source Nodes: [sp_187], Original ATen: [aten.convolution]
        buf861 = extern_kernels.convolution(buf860, buf46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf861, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf862 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_187], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf861, buf862, 224, 784, grid=grid(224, 784), stream=stream0)
        buf863 = buf854; del buf854  # reuse
        buf864 = buf853; del buf853  # reuse
        buf865 = buf852; del buf852  # reuse
        # Source Nodes: [sp_188], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf862, buf863, buf864, buf865, 1372, 128, grid=grid(1372), stream=stream0)
        buf866 = buf856; del buf856  # reuse
        buf867 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf869 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_188], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf863, buf864, buf865, primals_633, primals_634, buf866, buf867, buf869, primals_633, primals_634, 28, 49, grid=grid(28), stream=stream0)
        del primals_633
        del primals_634
        buf870 = reinterpret_tensor(buf905, (8, 28, 28, 28), (175616, 784, 28, 1), 65856)  # alias
        buf1895 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_188, sp_189], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf862, buf866, buf867, primals_185, primals_186, buf870, buf1895, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_186
        buf871 = reinterpret_tensor(buf861, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf861  # reuse
        # Source Nodes: [sp_190], Original ATen: [aten.add]
        triton_poi_fused_add_57.run(buf870, buf827, buf871, 6272, 28, grid=grid(6272, 28), stream=stream0)
        # Source Nodes: [sp_191], Original ATen: [aten.convolution]
        buf872 = extern_kernels.convolution(buf871, buf47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf872, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf873 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_191], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf872, buf873, 224, 784, grid=grid(224, 784), stream=stream0)
        buf874 = buf865; del buf865  # reuse
        buf875 = buf864; del buf864  # reuse
        buf876 = buf863; del buf863  # reuse
        # Source Nodes: [sp_192], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf873, buf874, buf875, buf876, 1372, 128, grid=grid(1372), stream=stream0)
        buf877 = buf867; del buf867  # reuse
        buf878 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf880 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_192], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf874, buf875, buf876, primals_636, primals_637, buf877, buf878, buf880, primals_636, primals_637, 28, 49, grid=grid(28), stream=stream0)
        del primals_636
        del primals_637
        buf881 = reinterpret_tensor(buf905, (8, 28, 28, 28), (175616, 784, 28, 1), 87808)  # alias
        buf1894 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_192, sp_193], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf873, buf877, buf878, primals_188, primals_189, buf881, buf1894, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_189
        buf882 = reinterpret_tensor(buf872, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf872  # reuse
        # Source Nodes: [sp_194], Original ATen: [aten.add]
        triton_poi_fused_add_58.run(buf881, buf827, buf882, 6272, 28, grid=grid(6272, 28), stream=stream0)
        # Source Nodes: [sp_195], Original ATen: [aten.convolution]
        buf883 = extern_kernels.convolution(buf882, buf48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf883, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf884 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_195], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf883, buf884, 224, 784, grid=grid(224, 784), stream=stream0)
        buf885 = buf876; del buf876  # reuse
        buf886 = buf875; del buf875  # reuse
        buf887 = buf874; del buf874  # reuse
        # Source Nodes: [sp_196], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf884, buf885, buf886, buf887, 1372, 128, grid=grid(1372), stream=stream0)
        buf888 = buf878; del buf878  # reuse
        buf889 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf891 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_196], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf885, buf886, buf887, primals_639, primals_640, buf888, buf889, buf891, primals_639, primals_640, 28, 49, grid=grid(28), stream=stream0)
        del primals_639
        del primals_640
        buf892 = reinterpret_tensor(buf905, (8, 28, 28, 28), (175616, 784, 28, 1), 109760)  # alias
        buf1893 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_196, sp_197], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf884, buf888, buf889, primals_191, primals_192, buf892, buf1893, 224, 784, grid=grid(224, 784), stream=stream0)
        del primals_192
        buf893 = reinterpret_tensor(buf883, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf883  # reuse
        # Source Nodes: [sp_198], Original ATen: [aten.add]
        triton_poi_fused_add_59.run(buf892, buf827, buf893, 6272, 28, grid=grid(6272, 28), stream=stream0)
        # Source Nodes: [sp_199], Original ATen: [aten.convolution]
        buf894 = extern_kernels.convolution(buf893, buf49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf894, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf895 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_199], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf894, buf895, 224, 784, grid=grid(224, 784), stream=stream0)
        del buf894
        buf896 = buf887; del buf887  # reuse
        buf897 = buf886; del buf886  # reuse
        buf898 = buf885; del buf885  # reuse
        # Source Nodes: [sp_200], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf895, buf896, buf897, buf898, 1372, 128, grid=grid(1372), stream=stream0)
        buf899 = buf889; del buf889  # reuse
        buf900 = empty_strided((1, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        buf902 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_200], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf896, buf897, buf898, primals_642, primals_643, buf899, buf900, buf902, primals_642, primals_643, 28, 49, grid=grid(28), stream=stream0)
        del buf896
        del buf897
        del buf898
        del primals_642
        del primals_643
        buf903 = reinterpret_tensor(buf905, (8, 28, 28, 28), (175616, 784, 28, 1), 131712)  # alias
        buf1892 = empty_strided((8, 28, 28, 28), (21952, 1, 784, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_200, sp_201], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf895, buf899, buf900, primals_194, primals_195, buf903, buf1892, 224, 784, grid=grid(224, 784), stream=stream0)
        del buf900
        del primals_195
        buf904 = reinterpret_tensor(buf905, (8, 28, 28, 28), (175616, 784, 28, 1), 153664)  # alias
        # Source Nodes: [cat_25], Original ATen: [aten.cat]
        triton_poi_fused_cat_60.run(buf827, buf904, 224, 784, grid=grid(224, 784), stream=stream0)
        buf906 = empty_strided((8, 224, 28, 28), (175616, 1, 6272, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_25], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf905, buf906, 1792, 784, grid=grid(1792, 784), stream=stream0)
        del buf837
        del buf848
        del buf859
        del buf870
        del buf881
        del buf892
        del buf903
        del buf904
        # Source Nodes: [out_52], Original ATen: [aten.convolution]
        buf907 = extern_kernels.convolution(buf906, primals_196, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf907, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf908 = empty_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_52], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf907, buf908, 4096, 784, grid=grid(4096, 784), stream=stream0)
        buf909 = buf812; del buf812  # reuse
        buf910 = buf811; del buf811  # reuse
        buf911 = buf810; del buf810  # reuse
        # Source Nodes: [out_53], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_48.run(buf908, buf909, buf910, buf911, 25088, 128, grid=grid(25088), stream=stream0)
        buf912 = buf814; del buf814  # reuse
        buf913 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf915 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_53], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_49.run(buf909, buf910, buf911, primals_645, primals_646, buf912, buf913, buf915, primals_645, primals_646, 512, 49, grid=grid(512), stream=stream0)
        del buf909
        del buf910
        del buf911
        del primals_645
        del primals_646
        buf916 = reinterpret_tensor(buf907, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf907  # reuse
        # Source Nodes: [out_53, out_54, shortcut_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_61.run(buf908, buf912, buf913, primals_197, primals_198, buf817, buf916, 3211264, grid=grid(3211264), stream=stream0)
        del buf913
        del primals_198
        # Source Nodes: [out_56], Original ATen: [aten.convolution]
        buf917 = extern_kernels.convolution(buf916, primals_199, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf917, (8, 448, 28, 28), (351232, 784, 28, 1))
        buf918 = reinterpret_tensor(buf499, (8, 448, 28, 28), (351232, 1, 12544, 448), 0); del buf499  # reuse
        # Source Nodes: [out_56], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_62.run(buf917, buf918, 3584, 784, grid=grid(3584, 784), stream=stream0)
        buf919 = reinterpret_tensor(buf392, (1, 448, 1, 1, 49), (21952, 1, 21952, 21952, 448), 0); del buf392  # reuse
        buf920 = reinterpret_tensor(buf391, (1, 448, 1, 1, 49), (21952, 1, 21952, 21952, 448), 0); del buf391  # reuse
        buf921 = reinterpret_tensor(buf390, (1, 448, 1, 1, 49), (21952, 1, 21952, 21952, 448), 0); del buf390  # reuse
        # Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_63.run(buf918, buf919, buf920, buf921, 21952, 128, grid=grid(21952), stream=stream0)
        buf922 = reinterpret_tensor(buf521, (1, 448, 1, 1), (448, 1, 448, 448), 0); del buf521  # reuse
        buf923 = reinterpret_tensor(buf520, (1, 448, 1, 1), (448, 1, 448, 448), 0); del buf520  # reuse
        buf925 = reinterpret_tensor(buf519, (448, ), (1, ), 0); del buf519  # reuse
        # Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_64.run(buf919, buf920, buf921, primals_648, primals_649, buf922, buf923, buf925, primals_648, primals_649, 448, 49, grid=grid(448), stream=stream0)
        del buf919
        del buf920
        del buf921
        del primals_648
        del primals_649
        buf926 = reinterpret_tensor(buf917, (8, 448, 28, 28), (351232, 1, 12544, 448), 0); del buf917  # reuse
        buf1891 = empty_strided((8, 448, 28, 28), (351232, 1, 12544, 448), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_57, out_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_65.run(buf918, buf922, buf923, primals_200, primals_201, buf926, buf1891, 2809856, grid=grid(2809856), stream=stream0)
        del primals_201
        # Source Nodes: [sp_204], Original ATen: [aten.convolution]
        buf927 = extern_kernels.convolution(reinterpret_tensor(buf926, (8, 56, 28, 28), (351232, 1, 12544, 448), 0), buf50, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf927, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf928 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_204], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf927, buf928, 448, 196, grid=grid(448, 196), stream=stream0)
        buf929 = empty_strided((1, 56, 1, 1, 13), (728, 1, 728, 728, 56), device='cuda', dtype=torch.float32)
        buf930 = empty_strided((1, 56, 1, 1, 13), (728, 1, 728, 728, 56), device='cuda', dtype=torch.float32)
        buf931 = empty_strided((1, 56, 1, 1, 13), (728, 1, 728, 728, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf928, buf929, buf930, buf931, 728, 121, grid=grid(728), stream=stream0)
        buf932 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf933 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf935 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf929, buf930, buf931, primals_651, primals_652, buf932, buf933, buf935, primals_651, primals_652, 56, 13, grid=grid(56), stream=stream0)
        del primals_651
        del primals_652
        buf998 = empty((8, 448, 14, 14), device='cuda', dtype=torch.float32)
        buf936 = reinterpret_tensor(buf998, (8, 56, 14, 14), (87808, 196, 14, 1), 0)  # alias
        buf1890 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_205, sp_206], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf928, buf932, buf933, primals_203, primals_204, buf936, buf1890, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_204
        # Source Nodes: [sp_208], Original ATen: [aten.convolution]
        buf937 = extern_kernels.convolution(reinterpret_tensor(buf926, (8, 56, 28, 28), (351232, 1, 12544, 448), 56), buf51, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf937, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf938 = reinterpret_tensor(buf927, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf927  # reuse
        # Source Nodes: [sp_208], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf937, buf938, 448, 196, grid=grid(448, 196), stream=stream0)
        buf939 = buf931; del buf931  # reuse
        buf940 = buf930; del buf930  # reuse
        buf941 = buf929; del buf929  # reuse
        # Source Nodes: [sp_209], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf938, buf939, buf940, buf941, 728, 121, grid=grid(728), stream=stream0)
        buf942 = buf933; del buf933  # reuse
        buf943 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf945 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_209], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf939, buf940, buf941, primals_654, primals_655, buf942, buf943, buf945, primals_654, primals_655, 56, 13, grid=grid(56), stream=stream0)
        del primals_654
        del primals_655
        buf946 = reinterpret_tensor(buf998, (8, 56, 14, 14), (87808, 196, 14, 1), 10976)  # alias
        buf1889 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_209, sp_210], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf938, buf942, buf943, primals_206, primals_207, buf946, buf1889, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_207
        # Source Nodes: [sp_212], Original ATen: [aten.convolution]
        buf947 = extern_kernels.convolution(reinterpret_tensor(buf926, (8, 56, 28, 28), (351232, 1, 12544, 448), 112), buf52, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf947, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf948 = reinterpret_tensor(buf937, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf937  # reuse
        # Source Nodes: [sp_212], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf947, buf948, 448, 196, grid=grid(448, 196), stream=stream0)
        buf949 = buf941; del buf941  # reuse
        buf950 = buf940; del buf940  # reuse
        buf951 = buf939; del buf939  # reuse
        # Source Nodes: [sp_213], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf948, buf949, buf950, buf951, 728, 121, grid=grid(728), stream=stream0)
        buf952 = buf943; del buf943  # reuse
        buf953 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf955 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_213], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf949, buf950, buf951, primals_657, primals_658, buf952, buf953, buf955, primals_657, primals_658, 56, 13, grid=grid(56), stream=stream0)
        del primals_657
        del primals_658
        buf956 = reinterpret_tensor(buf998, (8, 56, 14, 14), (87808, 196, 14, 1), 21952)  # alias
        buf1888 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_213, sp_214], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf948, buf952, buf953, primals_209, primals_210, buf956, buf1888, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_210
        # Source Nodes: [sp_216], Original ATen: [aten.convolution]
        buf957 = extern_kernels.convolution(reinterpret_tensor(buf926, (8, 56, 28, 28), (351232, 1, 12544, 448), 168), buf53, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf957, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf958 = reinterpret_tensor(buf947, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf947  # reuse
        # Source Nodes: [sp_216], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf957, buf958, 448, 196, grid=grid(448, 196), stream=stream0)
        buf959 = buf951; del buf951  # reuse
        buf960 = buf950; del buf950  # reuse
        buf961 = buf949; del buf949  # reuse
        # Source Nodes: [sp_217], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf958, buf959, buf960, buf961, 728, 121, grid=grid(728), stream=stream0)
        buf962 = buf953; del buf953  # reuse
        buf963 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf965 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_217], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf959, buf960, buf961, primals_660, primals_661, buf962, buf963, buf965, primals_660, primals_661, 56, 13, grid=grid(56), stream=stream0)
        del primals_660
        del primals_661
        buf966 = reinterpret_tensor(buf998, (8, 56, 14, 14), (87808, 196, 14, 1), 32928)  # alias
        buf1887 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_217, sp_218], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf958, buf962, buf963, primals_212, primals_213, buf966, buf1887, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_213
        # Source Nodes: [sp_220], Original ATen: [aten.convolution]
        buf967 = extern_kernels.convolution(reinterpret_tensor(buf926, (8, 56, 28, 28), (351232, 1, 12544, 448), 224), buf54, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf967, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf968 = reinterpret_tensor(buf957, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf957  # reuse
        # Source Nodes: [sp_220], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf967, buf968, 448, 196, grid=grid(448, 196), stream=stream0)
        buf969 = buf961; del buf961  # reuse
        buf970 = buf960; del buf960  # reuse
        buf971 = buf959; del buf959  # reuse
        # Source Nodes: [sp_221], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf968, buf969, buf970, buf971, 728, 121, grid=grid(728), stream=stream0)
        buf972 = buf963; del buf963  # reuse
        buf973 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf975 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_221], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf969, buf970, buf971, primals_663, primals_664, buf972, buf973, buf975, primals_663, primals_664, 56, 13, grid=grid(56), stream=stream0)
        del primals_663
        del primals_664
        buf976 = reinterpret_tensor(buf998, (8, 56, 14, 14), (87808, 196, 14, 1), 43904)  # alias
        buf1886 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_221, sp_222], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf968, buf972, buf973, primals_215, primals_216, buf976, buf1886, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_216
        # Source Nodes: [sp_224], Original ATen: [aten.convolution]
        buf977 = extern_kernels.convolution(reinterpret_tensor(buf926, (8, 56, 28, 28), (351232, 1, 12544, 448), 280), buf55, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf977, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf978 = reinterpret_tensor(buf967, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf967  # reuse
        # Source Nodes: [sp_224], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf977, buf978, 448, 196, grid=grid(448, 196), stream=stream0)
        buf979 = buf971; del buf971  # reuse
        buf980 = buf970; del buf970  # reuse
        buf981 = buf969; del buf969  # reuse
        # Source Nodes: [sp_225], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf978, buf979, buf980, buf981, 728, 121, grid=grid(728), stream=stream0)
        buf982 = buf973; del buf973  # reuse
        buf983 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf985 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_225], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf979, buf980, buf981, primals_666, primals_667, buf982, buf983, buf985, primals_666, primals_667, 56, 13, grid=grid(56), stream=stream0)
        del primals_666
        del primals_667
        buf986 = reinterpret_tensor(buf998, (8, 56, 14, 14), (87808, 196, 14, 1), 54880)  # alias
        buf1885 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_225, sp_226], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf978, buf982, buf983, primals_218, primals_219, buf986, buf1885, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_219
        # Source Nodes: [sp_228], Original ATen: [aten.convolution]
        buf987 = extern_kernels.convolution(reinterpret_tensor(buf926, (8, 56, 28, 28), (351232, 1, 12544, 448), 336), buf56, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf987, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf988 = reinterpret_tensor(buf977, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf977  # reuse
        # Source Nodes: [sp_228], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf987, buf988, 448, 196, grid=grid(448, 196), stream=stream0)
        buf989 = buf981; del buf981  # reuse
        buf990 = buf980; del buf980  # reuse
        buf991 = buf979; del buf979  # reuse
        # Source Nodes: [sp_229], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf988, buf989, buf990, buf991, 728, 121, grid=grid(728), stream=stream0)
        buf992 = buf983; del buf983  # reuse
        buf993 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf995 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_229], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf989, buf990, buf991, primals_669, primals_670, buf992, buf993, buf995, primals_669, primals_670, 56, 13, grid=grid(56), stream=stream0)
        del primals_669
        del primals_670
        buf996 = reinterpret_tensor(buf998, (8, 56, 14, 14), (87808, 196, 14, 1), 65856)  # alias
        buf1884 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_229, sp_230], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf988, buf992, buf993, primals_221, primals_222, buf996, buf1884, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_222
        buf997 = reinterpret_tensor(buf998, (8, 56, 14, 14), (87808, 196, 14, 1), 76832)  # alias
        # Source Nodes: [getattr_l__mod___layer3___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_70.run(buf926, buf997, 448, 196, grid=grid(448, 196), stream=stream0)
        buf999 = empty_strided((8, 448, 14, 14), (87808, 1, 6272, 448), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_24], Original ATen: [aten.cat]
        triton_poi_fused_cat_71.run(buf998, buf999, 3584, 196, grid=grid(3584, 196), stream=stream0)
        del buf936
        del buf946
        del buf956
        del buf966
        del buf976
        del buf986
        del buf996
        del buf997
        # Source Nodes: [out_60], Original ATen: [aten.convolution]
        buf1000 = extern_kernels.convolution(buf999, primals_223, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1000, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1001 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_60], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf1000, buf1001, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1002 = empty_strided((1, 1024, 1, 1, 13), (13312, 1, 13312, 13312, 1024), device='cuda', dtype=torch.float32)
        buf1003 = empty_strided((1, 1024, 1, 1, 13), (13312, 1, 13312, 13312, 1024), device='cuda', dtype=torch.float32)
        buf1004 = empty_strided((1, 1024, 1, 1, 13), (13312, 1, 13312, 13312, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_73.run(buf1001, buf1002, buf1003, buf1004, 13312, 121, grid=grid(13312), stream=stream0)
        buf1005 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1006 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1008 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_74.run(buf1002, buf1003, buf1004, primals_672, primals_673, buf1005, buf1006, buf1008, primals_672, primals_673, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_672
        del primals_673
        # Source Nodes: [getattr_l__mod___layer3___0___downsample_0], Original ATen: [aten.convolution]
        buf1009 = extern_kernels.convolution(buf916, primals_226, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1009, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1010 = reinterpret_tensor(buf1000, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1000  # reuse
        # Source Nodes: [getattr_l__mod___layer3___0___downsample_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf1009, buf1010, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1011 = buf1004; del buf1004  # reuse
        buf1012 = buf1003; del buf1003  # reuse
        buf1013 = buf1002; del buf1002  # reuse
        # Source Nodes: [shortcut_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_73.run(buf1010, buf1011, buf1012, buf1013, 13312, 121, grid=grid(13312), stream=stream0)
        buf1014 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1015 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1017 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_74.run(buf1011, buf1012, buf1013, primals_675, primals_676, buf1014, buf1015, buf1017, primals_675, primals_676, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_675
        del primals_676
        buf1018 = reinterpret_tensor(buf1009, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1009  # reuse
        buf1019 = buf1018; del buf1018  # reuse
        # Source Nodes: [out_61, out_62, shortcut_10, shortcut_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_75.run(buf1019, buf1001, buf1005, buf1006, primals_224, primals_225, buf1010, buf1014, buf1015, primals_227, primals_228, 1605632, grid=grid(1605632), stream=stream0)
        del primals_225
        del primals_228
        # Source Nodes: [out_64], Original ATen: [aten.convolution]
        buf1020 = extern_kernels.convolution(buf1019, primals_229, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1020, (8, 448, 14, 14), (87808, 196, 14, 1))
        buf1021 = reinterpret_tensor(buf998, (8, 448, 14, 14), (87808, 1, 6272, 448), 0); del buf998  # reuse
        # Source Nodes: [out_64], Original ATen: [aten.convolution]
        triton_poi_fused_cat_71.run(buf1020, buf1021, 3584, 196, grid=grid(3584, 196), stream=stream0)
        buf1022 = empty_strided((1, 448, 1, 1, 13), (5824, 1, 5824, 5824, 448), device='cuda', dtype=torch.float32)
        buf1023 = empty_strided((1, 448, 1, 1, 13), (5824, 1, 5824, 5824, 448), device='cuda', dtype=torch.float32)
        buf1024 = empty_strided((1, 448, 1, 1, 13), (5824, 1, 5824, 5824, 448), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_65], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf1021, buf1022, buf1023, buf1024, 5824, 121, grid=grid(5824), stream=stream0)
        buf1025 = buf923; del buf923  # reuse
        buf1026 = empty_strided((1, 448, 1, 1), (448, 1, 448, 448), device='cuda', dtype=torch.float32)
        buf1028 = empty((448, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_65], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_77.run(buf1022, buf1023, buf1024, primals_678, primals_679, buf1025, buf1026, buf1028, primals_678, primals_679, 448, 13, grid=grid(448), stream=stream0)
        del primals_678
        del primals_679
        buf1029 = reinterpret_tensor(buf1020, (8, 448, 14, 14), (87808, 1, 6272, 448), 0); del buf1020  # reuse
        buf1883 = empty_strided((8, 448, 14, 14), (87808, 1, 6272, 448), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_65, out_66], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_78.run(buf1021, buf1025, buf1026, primals_230, primals_231, buf1029, buf1883, 702464, grid=grid(702464), stream=stream0)
        del primals_231
        # Source Nodes: [sp_233], Original ATen: [aten.convolution]
        buf1030 = extern_kernels.convolution(reinterpret_tensor(buf1029, (8, 56, 14, 14), (87808, 1, 6272, 448), 0), buf57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1030, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1031 = reinterpret_tensor(buf987, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf987  # reuse
        # Source Nodes: [sp_233], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1030, buf1031, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1032 = buf991; del buf991  # reuse
        buf1033 = buf990; del buf990  # reuse
        buf1034 = buf989; del buf989  # reuse
        # Source Nodes: [sp_234], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1031, buf1032, buf1033, buf1034, 728, 121, grid=grid(728), stream=stream0)
        buf1035 = buf993; del buf993  # reuse
        buf1036 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1038 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_234], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1032, buf1033, buf1034, primals_681, primals_682, buf1035, buf1036, buf1038, primals_681, primals_682, 56, 13, grid=grid(56), stream=stream0)
        del primals_681
        del primals_682
        buf1107 = empty((8, 448, 14, 14), device='cuda', dtype=torch.float32)
        buf1039 = reinterpret_tensor(buf1107, (8, 56, 14, 14), (87808, 196, 14, 1), 0)  # alias
        buf1882 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_234, sp_235], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1031, buf1035, buf1036, primals_233, primals_234, buf1039, buf1882, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_234
        buf1040 = reinterpret_tensor(buf1030, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1030  # reuse
        # Source Nodes: [sp_236], Original ATen: [aten.add]
        triton_poi_fused_add_79.run(buf1039, buf1029, buf1040, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_237], Original ATen: [aten.convolution]
        buf1041 = extern_kernels.convolution(buf1040, buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1041, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1042 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_237], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1041, buf1042, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1043 = buf1034; del buf1034  # reuse
        buf1044 = buf1033; del buf1033  # reuse
        buf1045 = buf1032; del buf1032  # reuse
        # Source Nodes: [sp_238], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1042, buf1043, buf1044, buf1045, 728, 121, grid=grid(728), stream=stream0)
        buf1046 = buf1036; del buf1036  # reuse
        buf1047 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1049 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_238], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1043, buf1044, buf1045, primals_684, primals_685, buf1046, buf1047, buf1049, primals_684, primals_685, 56, 13, grid=grid(56), stream=stream0)
        del primals_684
        del primals_685
        buf1050 = reinterpret_tensor(buf1107, (8, 56, 14, 14), (87808, 196, 14, 1), 10976)  # alias
        buf1881 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_238, sp_239], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1042, buf1046, buf1047, primals_236, primals_237, buf1050, buf1881, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_237
        buf1051 = reinterpret_tensor(buf1041, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1041  # reuse
        # Source Nodes: [sp_240], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(buf1050, buf1029, buf1051, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_241], Original ATen: [aten.convolution]
        buf1052 = extern_kernels.convolution(buf1051, buf59, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1052, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1053 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_241], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1052, buf1053, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1054 = buf1045; del buf1045  # reuse
        buf1055 = buf1044; del buf1044  # reuse
        buf1056 = buf1043; del buf1043  # reuse
        # Source Nodes: [sp_242], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1053, buf1054, buf1055, buf1056, 728, 121, grid=grid(728), stream=stream0)
        buf1057 = buf1047; del buf1047  # reuse
        buf1058 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1060 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_242], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1054, buf1055, buf1056, primals_687, primals_688, buf1057, buf1058, buf1060, primals_687, primals_688, 56, 13, grid=grid(56), stream=stream0)
        del primals_687
        del primals_688
        buf1061 = reinterpret_tensor(buf1107, (8, 56, 14, 14), (87808, 196, 14, 1), 21952)  # alias
        buf1880 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_242, sp_243], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1053, buf1057, buf1058, primals_239, primals_240, buf1061, buf1880, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_240
        buf1062 = reinterpret_tensor(buf1052, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1052  # reuse
        # Source Nodes: [sp_244], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(buf1061, buf1029, buf1062, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_245], Original ATen: [aten.convolution]
        buf1063 = extern_kernels.convolution(buf1062, buf60, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1063, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1064 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_245], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1063, buf1064, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1065 = buf1056; del buf1056  # reuse
        buf1066 = buf1055; del buf1055  # reuse
        buf1067 = buf1054; del buf1054  # reuse
        # Source Nodes: [sp_246], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1064, buf1065, buf1066, buf1067, 728, 121, grid=grid(728), stream=stream0)
        buf1068 = buf1058; del buf1058  # reuse
        buf1069 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1071 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_246], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1065, buf1066, buf1067, primals_690, primals_691, buf1068, buf1069, buf1071, primals_690, primals_691, 56, 13, grid=grid(56), stream=stream0)
        del primals_690
        del primals_691
        buf1072 = reinterpret_tensor(buf1107, (8, 56, 14, 14), (87808, 196, 14, 1), 32928)  # alias
        buf1879 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_246, sp_247], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1064, buf1068, buf1069, primals_242, primals_243, buf1072, buf1879, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_243
        buf1073 = reinterpret_tensor(buf1063, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1063  # reuse
        # Source Nodes: [sp_248], Original ATen: [aten.add]
        triton_poi_fused_add_82.run(buf1072, buf1029, buf1073, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_249], Original ATen: [aten.convolution]
        buf1074 = extern_kernels.convolution(buf1073, buf61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1074, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1075 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_249], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1074, buf1075, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1076 = buf1067; del buf1067  # reuse
        buf1077 = buf1066; del buf1066  # reuse
        buf1078 = buf1065; del buf1065  # reuse
        # Source Nodes: [sp_250], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1075, buf1076, buf1077, buf1078, 728, 121, grid=grid(728), stream=stream0)
        buf1079 = buf1069; del buf1069  # reuse
        buf1080 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1082 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_250], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1076, buf1077, buf1078, primals_693, primals_694, buf1079, buf1080, buf1082, primals_693, primals_694, 56, 13, grid=grid(56), stream=stream0)
        del primals_693
        del primals_694
        buf1083 = reinterpret_tensor(buf1107, (8, 56, 14, 14), (87808, 196, 14, 1), 43904)  # alias
        buf1878 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_250, sp_251], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1075, buf1079, buf1080, primals_245, primals_246, buf1083, buf1878, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_246
        buf1084 = reinterpret_tensor(buf1074, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1074  # reuse
        # Source Nodes: [sp_252], Original ATen: [aten.add]
        triton_poi_fused_add_83.run(buf1083, buf1029, buf1084, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_253], Original ATen: [aten.convolution]
        buf1085 = extern_kernels.convolution(buf1084, buf62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1085, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1086 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_253], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1085, buf1086, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1087 = buf1078; del buf1078  # reuse
        buf1088 = buf1077; del buf1077  # reuse
        buf1089 = buf1076; del buf1076  # reuse
        # Source Nodes: [sp_254], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1086, buf1087, buf1088, buf1089, 728, 121, grid=grid(728), stream=stream0)
        buf1090 = buf1080; del buf1080  # reuse
        buf1091 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1093 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_254], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1087, buf1088, buf1089, primals_696, primals_697, buf1090, buf1091, buf1093, primals_696, primals_697, 56, 13, grid=grid(56), stream=stream0)
        del primals_696
        del primals_697
        buf1094 = reinterpret_tensor(buf1107, (8, 56, 14, 14), (87808, 196, 14, 1), 54880)  # alias
        buf1877 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_254, sp_255], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1086, buf1090, buf1091, primals_248, primals_249, buf1094, buf1877, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_249
        buf1095 = reinterpret_tensor(buf1085, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1085  # reuse
        # Source Nodes: [sp_256], Original ATen: [aten.add]
        triton_poi_fused_add_84.run(buf1094, buf1029, buf1095, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_257], Original ATen: [aten.convolution]
        buf1096 = extern_kernels.convolution(buf1095, buf63, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1096, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1097 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_257], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1096, buf1097, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1098 = buf1089; del buf1089  # reuse
        buf1099 = buf1088; del buf1088  # reuse
        buf1100 = buf1087; del buf1087  # reuse
        # Source Nodes: [sp_258], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1097, buf1098, buf1099, buf1100, 728, 121, grid=grid(728), stream=stream0)
        buf1101 = buf1091; del buf1091  # reuse
        buf1102 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1104 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_258], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1098, buf1099, buf1100, primals_699, primals_700, buf1101, buf1102, buf1104, primals_699, primals_700, 56, 13, grid=grid(56), stream=stream0)
        del primals_699
        del primals_700
        buf1105 = reinterpret_tensor(buf1107, (8, 56, 14, 14), (87808, 196, 14, 1), 65856)  # alias
        buf1876 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_258, sp_259], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1097, buf1101, buf1102, primals_251, primals_252, buf1105, buf1876, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_252
        buf1106 = reinterpret_tensor(buf1107, (8, 56, 14, 14), (87808, 196, 14, 1), 76832)  # alias
        # Source Nodes: [cat_23], Original ATen: [aten.cat]
        triton_poi_fused_cat_85.run(buf1029, buf1106, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1108 = empty_strided((8, 448, 14, 14), (87808, 1, 6272, 448), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_23], Original ATen: [aten.cat]
        triton_poi_fused_cat_71.run(buf1107, buf1108, 3584, 196, grid=grid(3584, 196), stream=stream0)
        del buf1039
        del buf1050
        del buf1061
        del buf1072
        del buf1083
        del buf1094
        del buf1105
        del buf1106
        # Source Nodes: [out_68], Original ATen: [aten.convolution]
        buf1109 = extern_kernels.convolution(buf1108, primals_253, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1109, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1110 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_68], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf1109, buf1110, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1111 = buf1013; del buf1013  # reuse
        buf1112 = buf1012; del buf1012  # reuse
        buf1113 = buf1011; del buf1011  # reuse
        # Source Nodes: [out_69], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_73.run(buf1110, buf1111, buf1112, buf1113, 13312, 121, grid=grid(13312), stream=stream0)
        buf1114 = buf1015; del buf1015  # reuse
        buf1115 = buf1006; del buf1006  # reuse
        buf1117 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_69], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_74.run(buf1111, buf1112, buf1113, primals_702, primals_703, buf1114, buf1115, buf1117, primals_702, primals_703, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_702
        del primals_703
        buf1118 = reinterpret_tensor(buf1109, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1109  # reuse
        # Source Nodes: [out_69, out_70, shortcut_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_86.run(buf1110, buf1114, buf1115, primals_254, primals_255, buf1019, buf1118, 1605632, grid=grid(1605632), stream=stream0)
        del primals_255
        # Source Nodes: [out_72], Original ATen: [aten.convolution]
        buf1119 = extern_kernels.convolution(buf1118, primals_256, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1119, (8, 448, 14, 14), (87808, 196, 14, 1))
        buf1120 = reinterpret_tensor(buf1107, (8, 448, 14, 14), (87808, 1, 6272, 448), 0); del buf1107  # reuse
        # Source Nodes: [out_72], Original ATen: [aten.convolution]
        triton_poi_fused_cat_71.run(buf1119, buf1120, 3584, 196, grid=grid(3584, 196), stream=stream0)
        buf1121 = buf1024; del buf1024  # reuse
        buf1122 = buf1023; del buf1023  # reuse
        buf1123 = buf1022; del buf1022  # reuse
        # Source Nodes: [out_73], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf1120, buf1121, buf1122, buf1123, 5824, 121, grid=grid(5824), stream=stream0)
        buf1124 = buf1026; del buf1026  # reuse
        buf1125 = empty_strided((1, 448, 1, 1), (448, 1, 448, 448), device='cuda', dtype=torch.float32)
        buf1127 = empty((448, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_73], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_77.run(buf1121, buf1122, buf1123, primals_705, primals_706, buf1124, buf1125, buf1127, primals_705, primals_706, 448, 13, grid=grid(448), stream=stream0)
        del primals_705
        del primals_706
        buf1128 = reinterpret_tensor(buf1119, (8, 448, 14, 14), (87808, 1, 6272, 448), 0); del buf1119  # reuse
        buf1875 = empty_strided((8, 448, 14, 14), (87808, 1, 6272, 448), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_73, out_74], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_78.run(buf1120, buf1124, buf1125, primals_257, primals_258, buf1128, buf1875, 702464, grid=grid(702464), stream=stream0)
        del primals_258
        # Source Nodes: [sp_262], Original ATen: [aten.convolution]
        buf1129 = extern_kernels.convolution(reinterpret_tensor(buf1128, (8, 56, 14, 14), (87808, 1, 6272, 448), 0), buf64, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1129, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1130 = reinterpret_tensor(buf1096, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1096  # reuse
        # Source Nodes: [sp_262], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1129, buf1130, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1131 = buf1100; del buf1100  # reuse
        buf1132 = buf1099; del buf1099  # reuse
        buf1133 = buf1098; del buf1098  # reuse
        # Source Nodes: [sp_263], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1130, buf1131, buf1132, buf1133, 728, 121, grid=grid(728), stream=stream0)
        buf1134 = buf1102; del buf1102  # reuse
        buf1135 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1137 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_263], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1131, buf1132, buf1133, primals_708, primals_709, buf1134, buf1135, buf1137, primals_708, primals_709, 56, 13, grid=grid(56), stream=stream0)
        del primals_708
        del primals_709
        buf1206 = empty((8, 448, 14, 14), device='cuda', dtype=torch.float32)
        buf1138 = reinterpret_tensor(buf1206, (8, 56, 14, 14), (87808, 196, 14, 1), 0)  # alias
        buf1874 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_263, sp_264], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1130, buf1134, buf1135, primals_260, primals_261, buf1138, buf1874, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_261
        buf1139 = reinterpret_tensor(buf1129, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1129  # reuse
        # Source Nodes: [sp_265], Original ATen: [aten.add]
        triton_poi_fused_add_79.run(buf1138, buf1128, buf1139, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_266], Original ATen: [aten.convolution]
        buf1140 = extern_kernels.convolution(buf1139, buf65, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1140, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1141 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_266], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1140, buf1141, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1142 = buf1133; del buf1133  # reuse
        buf1143 = buf1132; del buf1132  # reuse
        buf1144 = buf1131; del buf1131  # reuse
        # Source Nodes: [sp_267], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1141, buf1142, buf1143, buf1144, 728, 121, grid=grid(728), stream=stream0)
        buf1145 = buf1135; del buf1135  # reuse
        buf1146 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1148 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_267], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1142, buf1143, buf1144, primals_711, primals_712, buf1145, buf1146, buf1148, primals_711, primals_712, 56, 13, grid=grid(56), stream=stream0)
        del primals_711
        del primals_712
        buf1149 = reinterpret_tensor(buf1206, (8, 56, 14, 14), (87808, 196, 14, 1), 10976)  # alias
        buf1873 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_267, sp_268], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1141, buf1145, buf1146, primals_263, primals_264, buf1149, buf1873, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_264
        buf1150 = reinterpret_tensor(buf1140, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1140  # reuse
        # Source Nodes: [sp_269], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(buf1149, buf1128, buf1150, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_270], Original ATen: [aten.convolution]
        buf1151 = extern_kernels.convolution(buf1150, buf66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1151, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1152 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_270], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1151, buf1152, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1153 = buf1144; del buf1144  # reuse
        buf1154 = buf1143; del buf1143  # reuse
        buf1155 = buf1142; del buf1142  # reuse
        # Source Nodes: [sp_271], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1152, buf1153, buf1154, buf1155, 728, 121, grid=grid(728), stream=stream0)
        buf1156 = buf1146; del buf1146  # reuse
        buf1157 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1159 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_271], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1153, buf1154, buf1155, primals_714, primals_715, buf1156, buf1157, buf1159, primals_714, primals_715, 56, 13, grid=grid(56), stream=stream0)
        del primals_714
        del primals_715
        buf1160 = reinterpret_tensor(buf1206, (8, 56, 14, 14), (87808, 196, 14, 1), 21952)  # alias
        buf1872 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_271, sp_272], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1152, buf1156, buf1157, primals_266, primals_267, buf1160, buf1872, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_267
        buf1161 = reinterpret_tensor(buf1151, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1151  # reuse
        # Source Nodes: [sp_273], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(buf1160, buf1128, buf1161, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_274], Original ATen: [aten.convolution]
        buf1162 = extern_kernels.convolution(buf1161, buf67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1162, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1163 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_274], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1162, buf1163, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1164 = buf1155; del buf1155  # reuse
        buf1165 = buf1154; del buf1154  # reuse
        buf1166 = buf1153; del buf1153  # reuse
        # Source Nodes: [sp_275], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1163, buf1164, buf1165, buf1166, 728, 121, grid=grid(728), stream=stream0)
        buf1167 = buf1157; del buf1157  # reuse
        buf1168 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1170 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_275], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1164, buf1165, buf1166, primals_717, primals_718, buf1167, buf1168, buf1170, primals_717, primals_718, 56, 13, grid=grid(56), stream=stream0)
        del primals_717
        del primals_718
        buf1171 = reinterpret_tensor(buf1206, (8, 56, 14, 14), (87808, 196, 14, 1), 32928)  # alias
        buf1871 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_275, sp_276], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1163, buf1167, buf1168, primals_269, primals_270, buf1171, buf1871, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_270
        buf1172 = reinterpret_tensor(buf1162, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1162  # reuse
        # Source Nodes: [sp_277], Original ATen: [aten.add]
        triton_poi_fused_add_82.run(buf1171, buf1128, buf1172, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_278], Original ATen: [aten.convolution]
        buf1173 = extern_kernels.convolution(buf1172, buf68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1173, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1174 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_278], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1173, buf1174, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1175 = buf1166; del buf1166  # reuse
        buf1176 = buf1165; del buf1165  # reuse
        buf1177 = buf1164; del buf1164  # reuse
        # Source Nodes: [sp_279], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1174, buf1175, buf1176, buf1177, 728, 121, grid=grid(728), stream=stream0)
        buf1178 = buf1168; del buf1168  # reuse
        buf1179 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1181 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_279], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1175, buf1176, buf1177, primals_720, primals_721, buf1178, buf1179, buf1181, primals_720, primals_721, 56, 13, grid=grid(56), stream=stream0)
        del primals_720
        del primals_721
        buf1182 = reinterpret_tensor(buf1206, (8, 56, 14, 14), (87808, 196, 14, 1), 43904)  # alias
        buf1870 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_279, sp_280], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1174, buf1178, buf1179, primals_272, primals_273, buf1182, buf1870, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_273
        buf1183 = reinterpret_tensor(buf1173, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1173  # reuse
        # Source Nodes: [sp_281], Original ATen: [aten.add]
        triton_poi_fused_add_83.run(buf1182, buf1128, buf1183, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_282], Original ATen: [aten.convolution]
        buf1184 = extern_kernels.convolution(buf1183, buf69, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1184, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1185 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_282], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1184, buf1185, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1186 = buf1177; del buf1177  # reuse
        buf1187 = buf1176; del buf1176  # reuse
        buf1188 = buf1175; del buf1175  # reuse
        # Source Nodes: [sp_283], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1185, buf1186, buf1187, buf1188, 728, 121, grid=grid(728), stream=stream0)
        buf1189 = buf1179; del buf1179  # reuse
        buf1190 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1192 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_283], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1186, buf1187, buf1188, primals_723, primals_724, buf1189, buf1190, buf1192, primals_723, primals_724, 56, 13, grid=grid(56), stream=stream0)
        del primals_723
        del primals_724
        buf1193 = reinterpret_tensor(buf1206, (8, 56, 14, 14), (87808, 196, 14, 1), 54880)  # alias
        buf1869 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_283, sp_284], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1185, buf1189, buf1190, primals_275, primals_276, buf1193, buf1869, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_276
        buf1194 = reinterpret_tensor(buf1184, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1184  # reuse
        # Source Nodes: [sp_285], Original ATen: [aten.add]
        triton_poi_fused_add_84.run(buf1193, buf1128, buf1194, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_286], Original ATen: [aten.convolution]
        buf1195 = extern_kernels.convolution(buf1194, buf70, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1195, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1196 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_286], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1195, buf1196, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1197 = buf1188; del buf1188  # reuse
        buf1198 = buf1187; del buf1187  # reuse
        buf1199 = buf1186; del buf1186  # reuse
        # Source Nodes: [sp_287], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1196, buf1197, buf1198, buf1199, 728, 121, grid=grid(728), stream=stream0)
        buf1200 = buf1190; del buf1190  # reuse
        buf1201 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1203 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_287], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1197, buf1198, buf1199, primals_726, primals_727, buf1200, buf1201, buf1203, primals_726, primals_727, 56, 13, grid=grid(56), stream=stream0)
        del primals_726
        del primals_727
        buf1204 = reinterpret_tensor(buf1206, (8, 56, 14, 14), (87808, 196, 14, 1), 65856)  # alias
        buf1868 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_287, sp_288], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1196, buf1200, buf1201, primals_278, primals_279, buf1204, buf1868, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_279
        buf1205 = reinterpret_tensor(buf1206, (8, 56, 14, 14), (87808, 196, 14, 1), 76832)  # alias
        # Source Nodes: [cat_22], Original ATen: [aten.cat]
        triton_poi_fused_cat_85.run(buf1128, buf1205, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1207 = empty_strided((8, 448, 14, 14), (87808, 1, 6272, 448), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_22], Original ATen: [aten.cat]
        triton_poi_fused_cat_71.run(buf1206, buf1207, 3584, 196, grid=grid(3584, 196), stream=stream0)
        del buf1138
        del buf1149
        del buf1160
        del buf1171
        del buf1182
        del buf1193
        del buf1204
        del buf1205
        # Source Nodes: [out_76], Original ATen: [aten.convolution]
        buf1208 = extern_kernels.convolution(buf1207, primals_280, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1208, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1209 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_76], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf1208, buf1209, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1210 = buf1113; del buf1113  # reuse
        buf1211 = buf1112; del buf1112  # reuse
        buf1212 = buf1111; del buf1111  # reuse
        # Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_73.run(buf1209, buf1210, buf1211, buf1212, 13312, 121, grid=grid(13312), stream=stream0)
        buf1213 = buf1115; del buf1115  # reuse
        buf1214 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1216 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_74.run(buf1210, buf1211, buf1212, primals_729, primals_730, buf1213, buf1214, buf1216, primals_729, primals_730, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_729
        del primals_730
        buf1217 = reinterpret_tensor(buf1208, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1208  # reuse
        # Source Nodes: [out_77, out_78, shortcut_13], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_86.run(buf1209, buf1213, buf1214, primals_281, primals_282, buf1118, buf1217, 1605632, grid=grid(1605632), stream=stream0)
        del primals_282
        # Source Nodes: [out_80], Original ATen: [aten.convolution]
        buf1218 = extern_kernels.convolution(buf1217, primals_283, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1218, (8, 448, 14, 14), (87808, 196, 14, 1))
        buf1219 = reinterpret_tensor(buf1206, (8, 448, 14, 14), (87808, 1, 6272, 448), 0); del buf1206  # reuse
        # Source Nodes: [out_80], Original ATen: [aten.convolution]
        triton_poi_fused_cat_71.run(buf1218, buf1219, 3584, 196, grid=grid(3584, 196), stream=stream0)
        buf1220 = buf1123; del buf1123  # reuse
        buf1221 = buf1122; del buf1122  # reuse
        buf1222 = buf1121; del buf1121  # reuse
        # Source Nodes: [out_81], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf1219, buf1220, buf1221, buf1222, 5824, 121, grid=grid(5824), stream=stream0)
        buf1223 = buf1125; del buf1125  # reuse
        buf1224 = empty_strided((1, 448, 1, 1), (448, 1, 448, 448), device='cuda', dtype=torch.float32)
        buf1226 = empty((448, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_81], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_77.run(buf1220, buf1221, buf1222, primals_732, primals_733, buf1223, buf1224, buf1226, primals_732, primals_733, 448, 13, grid=grid(448), stream=stream0)
        del primals_732
        del primals_733
        buf1227 = reinterpret_tensor(buf1218, (8, 448, 14, 14), (87808, 1, 6272, 448), 0); del buf1218  # reuse
        buf1867 = empty_strided((8, 448, 14, 14), (87808, 1, 6272, 448), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_81, out_82], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_78.run(buf1219, buf1223, buf1224, primals_284, primals_285, buf1227, buf1867, 702464, grid=grid(702464), stream=stream0)
        del primals_285
        # Source Nodes: [sp_291], Original ATen: [aten.convolution]
        buf1228 = extern_kernels.convolution(reinterpret_tensor(buf1227, (8, 56, 14, 14), (87808, 1, 6272, 448), 0), buf71, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1228, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1229 = reinterpret_tensor(buf1195, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1195  # reuse
        # Source Nodes: [sp_291], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1228, buf1229, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1230 = buf1199; del buf1199  # reuse
        buf1231 = buf1198; del buf1198  # reuse
        buf1232 = buf1197; del buf1197  # reuse
        # Source Nodes: [sp_292], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1229, buf1230, buf1231, buf1232, 728, 121, grid=grid(728), stream=stream0)
        buf1233 = buf1201; del buf1201  # reuse
        buf1234 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1236 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_292], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1230, buf1231, buf1232, primals_735, primals_736, buf1233, buf1234, buf1236, primals_735, primals_736, 56, 13, grid=grid(56), stream=stream0)
        del primals_735
        del primals_736
        buf1305 = empty((8, 448, 14, 14), device='cuda', dtype=torch.float32)
        buf1237 = reinterpret_tensor(buf1305, (8, 56, 14, 14), (87808, 196, 14, 1), 0)  # alias
        buf1866 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_292, sp_293], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1229, buf1233, buf1234, primals_287, primals_288, buf1237, buf1866, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_288
        buf1238 = reinterpret_tensor(buf1228, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1228  # reuse
        # Source Nodes: [sp_294], Original ATen: [aten.add]
        triton_poi_fused_add_79.run(buf1237, buf1227, buf1238, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_295], Original ATen: [aten.convolution]
        buf1239 = extern_kernels.convolution(buf1238, buf72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1239, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1240 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_295], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1239, buf1240, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1241 = buf1232; del buf1232  # reuse
        buf1242 = buf1231; del buf1231  # reuse
        buf1243 = buf1230; del buf1230  # reuse
        # Source Nodes: [sp_296], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1240, buf1241, buf1242, buf1243, 728, 121, grid=grid(728), stream=stream0)
        buf1244 = buf1234; del buf1234  # reuse
        buf1245 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1247 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_296], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1241, buf1242, buf1243, primals_738, primals_739, buf1244, buf1245, buf1247, primals_738, primals_739, 56, 13, grid=grid(56), stream=stream0)
        del primals_738
        del primals_739
        buf1248 = reinterpret_tensor(buf1305, (8, 56, 14, 14), (87808, 196, 14, 1), 10976)  # alias
        buf1865 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_296, sp_297], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1240, buf1244, buf1245, primals_290, primals_291, buf1248, buf1865, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_291
        buf1249 = reinterpret_tensor(buf1239, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1239  # reuse
        # Source Nodes: [sp_298], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(buf1248, buf1227, buf1249, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_299], Original ATen: [aten.convolution]
        buf1250 = extern_kernels.convolution(buf1249, buf73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1250, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1251 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_299], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1250, buf1251, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1252 = buf1243; del buf1243  # reuse
        buf1253 = buf1242; del buf1242  # reuse
        buf1254 = buf1241; del buf1241  # reuse
        # Source Nodes: [sp_300], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1251, buf1252, buf1253, buf1254, 728, 121, grid=grid(728), stream=stream0)
        buf1255 = buf1245; del buf1245  # reuse
        buf1256 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1258 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_300], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1252, buf1253, buf1254, primals_741, primals_742, buf1255, buf1256, buf1258, primals_741, primals_742, 56, 13, grid=grid(56), stream=stream0)
        del primals_741
        del primals_742
        buf1259 = reinterpret_tensor(buf1305, (8, 56, 14, 14), (87808, 196, 14, 1), 21952)  # alias
        buf1864 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_300, sp_301], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1251, buf1255, buf1256, primals_293, primals_294, buf1259, buf1864, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_294
        buf1260 = reinterpret_tensor(buf1250, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1250  # reuse
        # Source Nodes: [sp_302], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(buf1259, buf1227, buf1260, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_303], Original ATen: [aten.convolution]
        buf1261 = extern_kernels.convolution(buf1260, buf74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1261, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1262 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_303], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1261, buf1262, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1263 = buf1254; del buf1254  # reuse
        buf1264 = buf1253; del buf1253  # reuse
        buf1265 = buf1252; del buf1252  # reuse
        # Source Nodes: [sp_304], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1262, buf1263, buf1264, buf1265, 728, 121, grid=grid(728), stream=stream0)
        buf1266 = buf1256; del buf1256  # reuse
        buf1267 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1269 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_304], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1263, buf1264, buf1265, primals_744, primals_745, buf1266, buf1267, buf1269, primals_744, primals_745, 56, 13, grid=grid(56), stream=stream0)
        del primals_744
        del primals_745
        buf1270 = reinterpret_tensor(buf1305, (8, 56, 14, 14), (87808, 196, 14, 1), 32928)  # alias
        buf1863 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_304, sp_305], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1262, buf1266, buf1267, primals_296, primals_297, buf1270, buf1863, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_297
        buf1271 = reinterpret_tensor(buf1261, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1261  # reuse
        # Source Nodes: [sp_306], Original ATen: [aten.add]
        triton_poi_fused_add_82.run(buf1270, buf1227, buf1271, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_307], Original ATen: [aten.convolution]
        buf1272 = extern_kernels.convolution(buf1271, buf75, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1272, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1273 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_307], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1272, buf1273, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1274 = buf1265; del buf1265  # reuse
        buf1275 = buf1264; del buf1264  # reuse
        buf1276 = buf1263; del buf1263  # reuse
        # Source Nodes: [sp_308], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1273, buf1274, buf1275, buf1276, 728, 121, grid=grid(728), stream=stream0)
        buf1277 = buf1267; del buf1267  # reuse
        buf1278 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1280 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_308], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1274, buf1275, buf1276, primals_747, primals_748, buf1277, buf1278, buf1280, primals_747, primals_748, 56, 13, grid=grid(56), stream=stream0)
        del primals_747
        del primals_748
        buf1281 = reinterpret_tensor(buf1305, (8, 56, 14, 14), (87808, 196, 14, 1), 43904)  # alias
        buf1862 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_308, sp_309], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1273, buf1277, buf1278, primals_299, primals_300, buf1281, buf1862, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_300
        buf1282 = reinterpret_tensor(buf1272, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1272  # reuse
        # Source Nodes: [sp_310], Original ATen: [aten.add]
        triton_poi_fused_add_83.run(buf1281, buf1227, buf1282, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_311], Original ATen: [aten.convolution]
        buf1283 = extern_kernels.convolution(buf1282, buf76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1283, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1284 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_311], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1283, buf1284, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1285 = buf1276; del buf1276  # reuse
        buf1286 = buf1275; del buf1275  # reuse
        buf1287 = buf1274; del buf1274  # reuse
        # Source Nodes: [sp_312], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1284, buf1285, buf1286, buf1287, 728, 121, grid=grid(728), stream=stream0)
        buf1288 = buf1278; del buf1278  # reuse
        buf1289 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1291 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_312], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1285, buf1286, buf1287, primals_750, primals_751, buf1288, buf1289, buf1291, primals_750, primals_751, 56, 13, grid=grid(56), stream=stream0)
        del primals_750
        del primals_751
        buf1292 = reinterpret_tensor(buf1305, (8, 56, 14, 14), (87808, 196, 14, 1), 54880)  # alias
        buf1861 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_312, sp_313], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1284, buf1288, buf1289, primals_302, primals_303, buf1292, buf1861, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_303
        buf1293 = reinterpret_tensor(buf1283, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1283  # reuse
        # Source Nodes: [sp_314], Original ATen: [aten.add]
        triton_poi_fused_add_84.run(buf1292, buf1227, buf1293, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_315], Original ATen: [aten.convolution]
        buf1294 = extern_kernels.convolution(buf1293, buf77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1294, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1295 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_315], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1294, buf1295, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1296 = buf1287; del buf1287  # reuse
        buf1297 = buf1286; del buf1286  # reuse
        buf1298 = buf1285; del buf1285  # reuse
        # Source Nodes: [sp_316], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1295, buf1296, buf1297, buf1298, 728, 121, grid=grid(728), stream=stream0)
        buf1299 = buf1289; del buf1289  # reuse
        buf1300 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1302 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_316], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1296, buf1297, buf1298, primals_753, primals_754, buf1299, buf1300, buf1302, primals_753, primals_754, 56, 13, grid=grid(56), stream=stream0)
        del primals_753
        del primals_754
        buf1303 = reinterpret_tensor(buf1305, (8, 56, 14, 14), (87808, 196, 14, 1), 65856)  # alias
        buf1860 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_316, sp_317], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1295, buf1299, buf1300, primals_305, primals_306, buf1303, buf1860, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_306
        buf1304 = reinterpret_tensor(buf1305, (8, 56, 14, 14), (87808, 196, 14, 1), 76832)  # alias
        # Source Nodes: [cat_21], Original ATen: [aten.cat]
        triton_poi_fused_cat_85.run(buf1227, buf1304, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1306 = empty_strided((8, 448, 14, 14), (87808, 1, 6272, 448), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_21], Original ATen: [aten.cat]
        triton_poi_fused_cat_71.run(buf1305, buf1306, 3584, 196, grid=grid(3584, 196), stream=stream0)
        del buf1237
        del buf1248
        del buf1259
        del buf1270
        del buf1281
        del buf1292
        del buf1303
        del buf1304
        # Source Nodes: [out_84], Original ATen: [aten.convolution]
        buf1307 = extern_kernels.convolution(buf1306, primals_307, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1307, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1308 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_84], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf1307, buf1308, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1309 = buf1212; del buf1212  # reuse
        buf1310 = buf1211; del buf1211  # reuse
        buf1311 = buf1210; del buf1210  # reuse
        # Source Nodes: [out_85], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_73.run(buf1308, buf1309, buf1310, buf1311, 13312, 121, grid=grid(13312), stream=stream0)
        buf1312 = buf1214; del buf1214  # reuse
        buf1313 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1315 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_85], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_74.run(buf1309, buf1310, buf1311, primals_756, primals_757, buf1312, buf1313, buf1315, primals_756, primals_757, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_756
        del primals_757
        buf1316 = reinterpret_tensor(buf1307, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1307  # reuse
        # Source Nodes: [out_85, out_86, shortcut_14], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_86.run(buf1308, buf1312, buf1313, primals_308, primals_309, buf1217, buf1316, 1605632, grid=grid(1605632), stream=stream0)
        del primals_309
        # Source Nodes: [out_88], Original ATen: [aten.convolution]
        buf1317 = extern_kernels.convolution(buf1316, primals_310, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1317, (8, 448, 14, 14), (87808, 196, 14, 1))
        buf1318 = reinterpret_tensor(buf1305, (8, 448, 14, 14), (87808, 1, 6272, 448), 0); del buf1305  # reuse
        # Source Nodes: [out_88], Original ATen: [aten.convolution]
        triton_poi_fused_cat_71.run(buf1317, buf1318, 3584, 196, grid=grid(3584, 196), stream=stream0)
        buf1319 = buf1222; del buf1222  # reuse
        buf1320 = buf1221; del buf1221  # reuse
        buf1321 = buf1220; del buf1220  # reuse
        # Source Nodes: [out_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf1318, buf1319, buf1320, buf1321, 5824, 121, grid=grid(5824), stream=stream0)
        buf1322 = buf1224; del buf1224  # reuse
        buf1323 = empty_strided((1, 448, 1, 1), (448, 1, 448, 448), device='cuda', dtype=torch.float32)
        buf1325 = empty((448, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_77.run(buf1319, buf1320, buf1321, primals_759, primals_760, buf1322, buf1323, buf1325, primals_759, primals_760, 448, 13, grid=grid(448), stream=stream0)
        del primals_759
        del primals_760
        buf1326 = reinterpret_tensor(buf1317, (8, 448, 14, 14), (87808, 1, 6272, 448), 0); del buf1317  # reuse
        buf1859 = empty_strided((8, 448, 14, 14), (87808, 1, 6272, 448), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_89, out_90], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_78.run(buf1318, buf1322, buf1323, primals_311, primals_312, buf1326, buf1859, 702464, grid=grid(702464), stream=stream0)
        del primals_312
        # Source Nodes: [sp_320], Original ATen: [aten.convolution]
        buf1327 = extern_kernels.convolution(reinterpret_tensor(buf1326, (8, 56, 14, 14), (87808, 1, 6272, 448), 0), buf78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1327, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1328 = reinterpret_tensor(buf1294, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1294  # reuse
        # Source Nodes: [sp_320], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1327, buf1328, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1329 = buf1298; del buf1298  # reuse
        buf1330 = buf1297; del buf1297  # reuse
        buf1331 = buf1296; del buf1296  # reuse
        # Source Nodes: [sp_321], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1328, buf1329, buf1330, buf1331, 728, 121, grid=grid(728), stream=stream0)
        buf1332 = buf1300; del buf1300  # reuse
        buf1333 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1335 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_321], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1329, buf1330, buf1331, primals_762, primals_763, buf1332, buf1333, buf1335, primals_762, primals_763, 56, 13, grid=grid(56), stream=stream0)
        del primals_762
        del primals_763
        buf1404 = empty((8, 448, 14, 14), device='cuda', dtype=torch.float32)
        buf1336 = reinterpret_tensor(buf1404, (8, 56, 14, 14), (87808, 196, 14, 1), 0)  # alias
        buf1858 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_321, sp_322], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1328, buf1332, buf1333, primals_314, primals_315, buf1336, buf1858, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_315
        buf1337 = reinterpret_tensor(buf1327, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1327  # reuse
        # Source Nodes: [sp_323], Original ATen: [aten.add]
        triton_poi_fused_add_79.run(buf1336, buf1326, buf1337, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_324], Original ATen: [aten.convolution]
        buf1338 = extern_kernels.convolution(buf1337, buf79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1338, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1339 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_324], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1338, buf1339, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1340 = buf1331; del buf1331  # reuse
        buf1341 = buf1330; del buf1330  # reuse
        buf1342 = buf1329; del buf1329  # reuse
        # Source Nodes: [sp_325], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1339, buf1340, buf1341, buf1342, 728, 121, grid=grid(728), stream=stream0)
        buf1343 = buf1333; del buf1333  # reuse
        buf1344 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1346 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_325], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1340, buf1341, buf1342, primals_765, primals_766, buf1343, buf1344, buf1346, primals_765, primals_766, 56, 13, grid=grid(56), stream=stream0)
        del primals_765
        del primals_766
        buf1347 = reinterpret_tensor(buf1404, (8, 56, 14, 14), (87808, 196, 14, 1), 10976)  # alias
        buf1857 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_325, sp_326], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1339, buf1343, buf1344, primals_317, primals_318, buf1347, buf1857, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_318
        buf1348 = reinterpret_tensor(buf1338, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1338  # reuse
        # Source Nodes: [sp_327], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(buf1347, buf1326, buf1348, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_328], Original ATen: [aten.convolution]
        buf1349 = extern_kernels.convolution(buf1348, buf80, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1349, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1350 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_328], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1349, buf1350, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1351 = buf1342; del buf1342  # reuse
        buf1352 = buf1341; del buf1341  # reuse
        buf1353 = buf1340; del buf1340  # reuse
        # Source Nodes: [sp_329], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1350, buf1351, buf1352, buf1353, 728, 121, grid=grid(728), stream=stream0)
        buf1354 = buf1344; del buf1344  # reuse
        buf1355 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1357 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_329], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1351, buf1352, buf1353, primals_768, primals_769, buf1354, buf1355, buf1357, primals_768, primals_769, 56, 13, grid=grid(56), stream=stream0)
        del primals_768
        del primals_769
        buf1358 = reinterpret_tensor(buf1404, (8, 56, 14, 14), (87808, 196, 14, 1), 21952)  # alias
        buf1856 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_329, sp_330], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1350, buf1354, buf1355, primals_320, primals_321, buf1358, buf1856, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_321
        buf1359 = reinterpret_tensor(buf1349, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1349  # reuse
        # Source Nodes: [sp_331], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(buf1358, buf1326, buf1359, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_332], Original ATen: [aten.convolution]
        buf1360 = extern_kernels.convolution(buf1359, buf81, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1360, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1361 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_332], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1360, buf1361, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1362 = buf1353; del buf1353  # reuse
        buf1363 = buf1352; del buf1352  # reuse
        buf1364 = buf1351; del buf1351  # reuse
        # Source Nodes: [sp_333], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1361, buf1362, buf1363, buf1364, 728, 121, grid=grid(728), stream=stream0)
        buf1365 = buf1355; del buf1355  # reuse
        buf1366 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1368 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_333], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1362, buf1363, buf1364, primals_771, primals_772, buf1365, buf1366, buf1368, primals_771, primals_772, 56, 13, grid=grid(56), stream=stream0)
        del primals_771
        del primals_772
        buf1369 = reinterpret_tensor(buf1404, (8, 56, 14, 14), (87808, 196, 14, 1), 32928)  # alias
        buf1855 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_333, sp_334], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1361, buf1365, buf1366, primals_323, primals_324, buf1369, buf1855, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_324
        buf1370 = reinterpret_tensor(buf1360, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1360  # reuse
        # Source Nodes: [sp_335], Original ATen: [aten.add]
        triton_poi_fused_add_82.run(buf1369, buf1326, buf1370, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_336], Original ATen: [aten.convolution]
        buf1371 = extern_kernels.convolution(buf1370, buf82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1371, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1372 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_336], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1371, buf1372, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1373 = buf1364; del buf1364  # reuse
        buf1374 = buf1363; del buf1363  # reuse
        buf1375 = buf1362; del buf1362  # reuse
        # Source Nodes: [sp_337], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1372, buf1373, buf1374, buf1375, 728, 121, grid=grid(728), stream=stream0)
        buf1376 = buf1366; del buf1366  # reuse
        buf1377 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1379 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_337], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1373, buf1374, buf1375, primals_774, primals_775, buf1376, buf1377, buf1379, primals_774, primals_775, 56, 13, grid=grid(56), stream=stream0)
        del primals_774
        del primals_775
        buf1380 = reinterpret_tensor(buf1404, (8, 56, 14, 14), (87808, 196, 14, 1), 43904)  # alias
        buf1854 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_337, sp_338], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1372, buf1376, buf1377, primals_326, primals_327, buf1380, buf1854, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_327
        buf1381 = reinterpret_tensor(buf1371, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1371  # reuse
        # Source Nodes: [sp_339], Original ATen: [aten.add]
        triton_poi_fused_add_83.run(buf1380, buf1326, buf1381, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_340], Original ATen: [aten.convolution]
        buf1382 = extern_kernels.convolution(buf1381, buf83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1382, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1383 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_340], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1382, buf1383, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1384 = buf1375; del buf1375  # reuse
        buf1385 = buf1374; del buf1374  # reuse
        buf1386 = buf1373; del buf1373  # reuse
        # Source Nodes: [sp_341], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1383, buf1384, buf1385, buf1386, 728, 121, grid=grid(728), stream=stream0)
        buf1387 = buf1377; del buf1377  # reuse
        buf1388 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1390 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_341], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1384, buf1385, buf1386, primals_777, primals_778, buf1387, buf1388, buf1390, primals_777, primals_778, 56, 13, grid=grid(56), stream=stream0)
        del primals_777
        del primals_778
        buf1391 = reinterpret_tensor(buf1404, (8, 56, 14, 14), (87808, 196, 14, 1), 54880)  # alias
        buf1853 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_341, sp_342], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1383, buf1387, buf1388, primals_329, primals_330, buf1391, buf1853, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_330
        buf1392 = reinterpret_tensor(buf1382, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1382  # reuse
        # Source Nodes: [sp_343], Original ATen: [aten.add]
        triton_poi_fused_add_84.run(buf1391, buf1326, buf1392, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_344], Original ATen: [aten.convolution]
        buf1393 = extern_kernels.convolution(buf1392, buf84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1393, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1394 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_344], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1393, buf1394, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1395 = buf1386; del buf1386  # reuse
        buf1396 = buf1385; del buf1385  # reuse
        buf1397 = buf1384; del buf1384  # reuse
        # Source Nodes: [sp_345], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1394, buf1395, buf1396, buf1397, 728, 121, grid=grid(728), stream=stream0)
        buf1398 = buf1388; del buf1388  # reuse
        buf1399 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1401 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_345], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1395, buf1396, buf1397, primals_780, primals_781, buf1398, buf1399, buf1401, primals_780, primals_781, 56, 13, grid=grid(56), stream=stream0)
        del primals_780
        del primals_781
        buf1402 = reinterpret_tensor(buf1404, (8, 56, 14, 14), (87808, 196, 14, 1), 65856)  # alias
        buf1852 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_345, sp_346], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1394, buf1398, buf1399, primals_332, primals_333, buf1402, buf1852, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_333
        buf1403 = reinterpret_tensor(buf1404, (8, 56, 14, 14), (87808, 196, 14, 1), 76832)  # alias
        # Source Nodes: [cat_20], Original ATen: [aten.cat]
        triton_poi_fused_cat_85.run(buf1326, buf1403, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1405 = empty_strided((8, 448, 14, 14), (87808, 1, 6272, 448), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_20], Original ATen: [aten.cat]
        triton_poi_fused_cat_71.run(buf1404, buf1405, 3584, 196, grid=grid(3584, 196), stream=stream0)
        del buf1336
        del buf1347
        del buf1358
        del buf1369
        del buf1380
        del buf1391
        del buf1402
        del buf1403
        # Source Nodes: [out_92], Original ATen: [aten.convolution]
        buf1406 = extern_kernels.convolution(buf1405, primals_334, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1406, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1407 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_92], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf1406, buf1407, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1408 = buf1311; del buf1311  # reuse
        buf1409 = buf1310; del buf1310  # reuse
        buf1410 = buf1309; del buf1309  # reuse
        # Source Nodes: [out_93], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_73.run(buf1407, buf1408, buf1409, buf1410, 13312, 121, grid=grid(13312), stream=stream0)
        buf1411 = buf1313; del buf1313  # reuse
        buf1412 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1414 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_93], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_74.run(buf1408, buf1409, buf1410, primals_783, primals_784, buf1411, buf1412, buf1414, primals_783, primals_784, 1024, 13, grid=grid(1024), stream=stream0)
        del primals_783
        del primals_784
        buf1415 = reinterpret_tensor(buf1406, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1406  # reuse
        # Source Nodes: [out_93, out_94, shortcut_15], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_86.run(buf1407, buf1411, buf1412, primals_335, primals_336, buf1316, buf1415, 1605632, grid=grid(1605632), stream=stream0)
        del primals_336
        # Source Nodes: [out_96], Original ATen: [aten.convolution]
        buf1416 = extern_kernels.convolution(buf1415, primals_337, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1416, (8, 448, 14, 14), (87808, 196, 14, 1))
        buf1417 = reinterpret_tensor(buf1404, (8, 448, 14, 14), (87808, 1, 6272, 448), 0); del buf1404  # reuse
        # Source Nodes: [out_96], Original ATen: [aten.convolution]
        triton_poi_fused_cat_71.run(buf1416, buf1417, 3584, 196, grid=grid(3584, 196), stream=stream0)
        buf1418 = buf1321; del buf1321  # reuse
        buf1419 = buf1320; del buf1320  # reuse
        buf1420 = buf1319; del buf1319  # reuse
        # Source Nodes: [out_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf1417, buf1418, buf1419, buf1420, 5824, 121, grid=grid(5824), stream=stream0)
        buf1421 = buf1323; del buf1323  # reuse
        buf1422 = empty_strided((1, 448, 1, 1), (448, 1, 448, 448), device='cuda', dtype=torch.float32)
        buf1424 = empty((448, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_77.run(buf1418, buf1419, buf1420, primals_786, primals_787, buf1421, buf1422, buf1424, primals_786, primals_787, 448, 13, grid=grid(448), stream=stream0)
        del buf1418
        del buf1419
        del buf1420
        del primals_786
        del primals_787
        buf1425 = reinterpret_tensor(buf1416, (8, 448, 14, 14), (87808, 1, 6272, 448), 0); del buf1416  # reuse
        buf1851 = empty_strided((8, 448, 14, 14), (87808, 1, 6272, 448), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_97, out_98], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_78.run(buf1417, buf1421, buf1422, primals_338, primals_339, buf1425, buf1851, 702464, grid=grid(702464), stream=stream0)
        del primals_339
        # Source Nodes: [sp_349], Original ATen: [aten.convolution]
        buf1426 = extern_kernels.convolution(reinterpret_tensor(buf1425, (8, 56, 14, 14), (87808, 1, 6272, 448), 0), buf85, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1426, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1427 = reinterpret_tensor(buf1393, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1393  # reuse
        # Source Nodes: [sp_349], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1426, buf1427, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1428 = buf1397; del buf1397  # reuse
        buf1429 = buf1396; del buf1396  # reuse
        buf1430 = buf1395; del buf1395  # reuse
        # Source Nodes: [sp_350], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1427, buf1428, buf1429, buf1430, 728, 121, grid=grid(728), stream=stream0)
        buf1431 = buf1399; del buf1399  # reuse
        buf1432 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1434 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_350], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1428, buf1429, buf1430, primals_789, primals_790, buf1431, buf1432, buf1434, primals_789, primals_790, 56, 13, grid=grid(56), stream=stream0)
        del primals_789
        del primals_790
        buf1503 = empty((8, 448, 14, 14), device='cuda', dtype=torch.float32)
        buf1435 = reinterpret_tensor(buf1503, (8, 56, 14, 14), (87808, 196, 14, 1), 0)  # alias
        buf1850 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_350, sp_351], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1427, buf1431, buf1432, primals_341, primals_342, buf1435, buf1850, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_342
        buf1436 = reinterpret_tensor(buf1426, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1426  # reuse
        # Source Nodes: [sp_352], Original ATen: [aten.add]
        triton_poi_fused_add_79.run(buf1435, buf1425, buf1436, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_353], Original ATen: [aten.convolution]
        buf1437 = extern_kernels.convolution(buf1436, buf86, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1437, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1438 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_353], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1437, buf1438, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1439 = buf1430; del buf1430  # reuse
        buf1440 = buf1429; del buf1429  # reuse
        buf1441 = buf1428; del buf1428  # reuse
        # Source Nodes: [sp_354], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1438, buf1439, buf1440, buf1441, 728, 121, grid=grid(728), stream=stream0)
        buf1442 = buf1432; del buf1432  # reuse
        buf1443 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1445 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_354], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1439, buf1440, buf1441, primals_792, primals_793, buf1442, buf1443, buf1445, primals_792, primals_793, 56, 13, grid=grid(56), stream=stream0)
        del primals_792
        del primals_793
        buf1446 = reinterpret_tensor(buf1503, (8, 56, 14, 14), (87808, 196, 14, 1), 10976)  # alias
        buf1849 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_354, sp_355], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1438, buf1442, buf1443, primals_344, primals_345, buf1446, buf1849, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_345
        buf1447 = reinterpret_tensor(buf1437, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1437  # reuse
        # Source Nodes: [sp_356], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(buf1446, buf1425, buf1447, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_357], Original ATen: [aten.convolution]
        buf1448 = extern_kernels.convolution(buf1447, buf87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1448, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1449 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_357], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1448, buf1449, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1450 = buf1441; del buf1441  # reuse
        buf1451 = buf1440; del buf1440  # reuse
        buf1452 = buf1439; del buf1439  # reuse
        # Source Nodes: [sp_358], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1449, buf1450, buf1451, buf1452, 728, 121, grid=grid(728), stream=stream0)
        buf1453 = buf1443; del buf1443  # reuse
        buf1454 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1456 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_358], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1450, buf1451, buf1452, primals_795, primals_796, buf1453, buf1454, buf1456, primals_795, primals_796, 56, 13, grid=grid(56), stream=stream0)
        del primals_795
        del primals_796
        buf1457 = reinterpret_tensor(buf1503, (8, 56, 14, 14), (87808, 196, 14, 1), 21952)  # alias
        buf1848 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_358, sp_359], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1449, buf1453, buf1454, primals_347, primals_348, buf1457, buf1848, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_348
        buf1458 = reinterpret_tensor(buf1448, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1448  # reuse
        # Source Nodes: [sp_360], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(buf1457, buf1425, buf1458, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_361], Original ATen: [aten.convolution]
        buf1459 = extern_kernels.convolution(buf1458, buf88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1459, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1460 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_361], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1459, buf1460, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1461 = buf1452; del buf1452  # reuse
        buf1462 = buf1451; del buf1451  # reuse
        buf1463 = buf1450; del buf1450  # reuse
        # Source Nodes: [sp_362], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1460, buf1461, buf1462, buf1463, 728, 121, grid=grid(728), stream=stream0)
        buf1464 = buf1454; del buf1454  # reuse
        buf1465 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1467 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_362], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1461, buf1462, buf1463, primals_798, primals_799, buf1464, buf1465, buf1467, primals_798, primals_799, 56, 13, grid=grid(56), stream=stream0)
        del primals_798
        del primals_799
        buf1468 = reinterpret_tensor(buf1503, (8, 56, 14, 14), (87808, 196, 14, 1), 32928)  # alias
        buf1847 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_362, sp_363], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1460, buf1464, buf1465, primals_350, primals_351, buf1468, buf1847, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_351
        buf1469 = reinterpret_tensor(buf1459, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1459  # reuse
        # Source Nodes: [sp_364], Original ATen: [aten.add]
        triton_poi_fused_add_82.run(buf1468, buf1425, buf1469, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_365], Original ATen: [aten.convolution]
        buf1470 = extern_kernels.convolution(buf1469, buf89, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1470, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1471 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_365], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1470, buf1471, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1472 = buf1463; del buf1463  # reuse
        buf1473 = buf1462; del buf1462  # reuse
        buf1474 = buf1461; del buf1461  # reuse
        # Source Nodes: [sp_366], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1471, buf1472, buf1473, buf1474, 728, 121, grid=grid(728), stream=stream0)
        buf1475 = buf1465; del buf1465  # reuse
        buf1476 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1478 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_366], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1472, buf1473, buf1474, primals_801, primals_802, buf1475, buf1476, buf1478, primals_801, primals_802, 56, 13, grid=grid(56), stream=stream0)
        del primals_801
        del primals_802
        buf1479 = reinterpret_tensor(buf1503, (8, 56, 14, 14), (87808, 196, 14, 1), 43904)  # alias
        buf1846 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_366, sp_367], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1471, buf1475, buf1476, primals_353, primals_354, buf1479, buf1846, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_354
        buf1480 = reinterpret_tensor(buf1470, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1470  # reuse
        # Source Nodes: [sp_368], Original ATen: [aten.add]
        triton_poi_fused_add_83.run(buf1479, buf1425, buf1480, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_369], Original ATen: [aten.convolution]
        buf1481 = extern_kernels.convolution(buf1480, buf90, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1481, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1482 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_369], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1481, buf1482, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1483 = buf1474; del buf1474  # reuse
        buf1484 = buf1473; del buf1473  # reuse
        buf1485 = buf1472; del buf1472  # reuse
        # Source Nodes: [sp_370], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1482, buf1483, buf1484, buf1485, 728, 121, grid=grid(728), stream=stream0)
        buf1486 = buf1476; del buf1476  # reuse
        buf1487 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1489 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_370], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1483, buf1484, buf1485, primals_804, primals_805, buf1486, buf1487, buf1489, primals_804, primals_805, 56, 13, grid=grid(56), stream=stream0)
        del primals_804
        del primals_805
        buf1490 = reinterpret_tensor(buf1503, (8, 56, 14, 14), (87808, 196, 14, 1), 54880)  # alias
        buf1845 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_370, sp_371], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1482, buf1486, buf1487, primals_356, primals_357, buf1490, buf1845, 448, 196, grid=grid(448, 196), stream=stream0)
        del primals_357
        buf1491 = reinterpret_tensor(buf1481, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf1481  # reuse
        # Source Nodes: [sp_372], Original ATen: [aten.add]
        triton_poi_fused_add_84.run(buf1490, buf1425, buf1491, 1568, 56, grid=grid(1568, 56), stream=stream0)
        # Source Nodes: [sp_373], Original ATen: [aten.convolution]
        buf1492 = extern_kernels.convolution(buf1491, buf91, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1492, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf1493 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_373], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf1492, buf1493, 448, 196, grid=grid(448, 196), stream=stream0)
        del buf1492
        buf1494 = buf1485; del buf1485  # reuse
        buf1495 = buf1484; del buf1484  # reuse
        buf1496 = buf1483; del buf1483  # reuse
        # Source Nodes: [sp_374], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf1493, buf1494, buf1495, buf1496, 728, 121, grid=grid(728), stream=stream0)
        buf1497 = buf1487; del buf1487  # reuse
        buf1498 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf1500 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_374], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf1494, buf1495, buf1496, primals_807, primals_808, buf1497, buf1498, buf1500, primals_807, primals_808, 56, 13, grid=grid(56), stream=stream0)
        del buf1494
        del buf1495
        del buf1496
        del primals_807
        del primals_808
        buf1501 = reinterpret_tensor(buf1503, (8, 56, 14, 14), (87808, 196, 14, 1), 65856)  # alias
        buf1844 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_374, sp_375], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_69.run(buf1493, buf1497, buf1498, primals_359, primals_360, buf1501, buf1844, 448, 196, grid=grid(448, 196), stream=stream0)
        del buf1498
        del primals_360
        buf1502 = reinterpret_tensor(buf1503, (8, 56, 14, 14), (87808, 196, 14, 1), 76832)  # alias
        # Source Nodes: [cat_19], Original ATen: [aten.cat]
        triton_poi_fused_cat_85.run(buf1425, buf1502, 448, 196, grid=grid(448, 196), stream=stream0)
        buf1504 = empty_strided((8, 448, 14, 14), (87808, 1, 6272, 448), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_19], Original ATen: [aten.cat]
        triton_poi_fused_cat_71.run(buf1503, buf1504, 3584, 196, grid=grid(3584, 196), stream=stream0)
        del buf1435
        del buf1446
        del buf1457
        del buf1468
        del buf1479
        del buf1490
        del buf1501
        del buf1502
        del buf1503
        # Source Nodes: [out_100], Original ATen: [aten.convolution]
        buf1505 = extern_kernels.convolution(buf1504, primals_361, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1505, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf1506 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_100], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf1505, buf1506, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf1507 = buf1410; del buf1410  # reuse
        buf1508 = buf1409; del buf1409  # reuse
        buf1509 = buf1408; del buf1408  # reuse
        # Source Nodes: [out_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_73.run(buf1506, buf1507, buf1508, buf1509, 13312, 121, grid=grid(13312), stream=stream0)
        buf1510 = buf1412; del buf1412  # reuse
        buf1511 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1513 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_74.run(buf1507, buf1508, buf1509, primals_810, primals_811, buf1510, buf1511, buf1513, primals_810, primals_811, 1024, 13, grid=grid(1024), stream=stream0)
        del buf1507
        del buf1508
        del buf1509
        del primals_810
        del primals_811
        buf1514 = reinterpret_tensor(buf1505, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1505  # reuse
        # Source Nodes: [out_101, out_102, shortcut_16], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_86.run(buf1506, buf1510, buf1511, primals_362, primals_363, buf1415, buf1514, 1605632, grid=grid(1605632), stream=stream0)
        del buf1511
        del primals_363
        # Source Nodes: [out_104], Original ATen: [aten.convolution]
        buf1515 = extern_kernels.convolution(buf1514, primals_364, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1515, (8, 896, 14, 14), (175616, 196, 14, 1))
        buf1516 = reinterpret_tensor(buf905, (8, 896, 14, 14), (175616, 1, 12544, 896), 0); del buf905  # reuse
        # Source Nodes: [out_104], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_87.run(buf1515, buf1516, 7168, 196, grid=grid(7168, 196), stream=stream0)
        buf1517 = empty_strided((1, 896, 1, 1, 13), (11648, 1, 11648, 11648, 896), device='cuda', dtype=torch.float32)
        buf1518 = empty_strided((1, 896, 1, 1, 13), (11648, 1, 11648, 11648, 896), device='cuda', dtype=torch.float32)
        buf1519 = empty_strided((1, 896, 1, 1, 13), (11648, 1, 11648, 11648, 896), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_105], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_88.run(buf1516, buf1517, buf1518, buf1519, 11648, 121, grid=grid(11648), stream=stream0)
        buf1520 = empty_strided((1, 896, 1, 1), (896, 1, 896, 896), device='cuda', dtype=torch.float32)
        buf1521 = empty_strided((1, 896, 1, 1), (896, 1, 896, 896), device='cuda', dtype=torch.float32)
        buf1523 = empty((896, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_105], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_89.run(buf1517, buf1518, buf1519, primals_813, primals_814, buf1520, buf1521, buf1523, primals_813, primals_814, 896, 13, grid=grid(896), stream=stream0)
        del buf1517
        del buf1518
        del buf1519
        del primals_813
        del primals_814
        buf1524 = reinterpret_tensor(buf1515, (8, 896, 14, 14), (175616, 1, 12544, 896), 0); del buf1515  # reuse
        buf1843 = empty_strided((8, 896, 14, 14), (175616, 1, 12544, 896), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_105, out_106], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_90.run(buf1516, buf1520, buf1521, primals_365, primals_366, buf1524, buf1843, 1404928, grid=grid(1404928), stream=stream0)
        del primals_366
        # Source Nodes: [sp_378], Original ATen: [aten.convolution]
        buf1525 = extern_kernels.convolution(reinterpret_tensor(buf1524, (8, 112, 14, 14), (175616, 1, 12544, 896), 0), buf92, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1525, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1526 = reinterpret_tensor(buf518, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf518  # reuse
        # Source Nodes: [sp_378], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1525, buf1526, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1527 = reinterpret_tensor(buf1422, (1, 112, 1, 1, 4), (448, 1, 448, 448, 112), 0); del buf1422  # reuse
        buf1528 = empty_strided((1, 112, 1, 1, 4), (448, 1, 448, 448, 112), device='cuda', dtype=torch.float32)
        buf1529 = empty_strided((1, 112, 1, 1, 4), (448, 1, 448, 448, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_379], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1526, buf1527, buf1528, buf1529, 448, 98, grid=grid(448), stream=stream0)
        buf1530 = buf397; del buf397  # reuse
        buf1531 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1533 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_379], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1527, buf1528, buf1529, primals_816, primals_817, buf1530, buf1531, buf1533, primals_816, primals_817, 112, 4, grid=grid(112), stream=stream0)
        del primals_816
        del primals_817
        buf1596 = reinterpret_tensor(buf485, (8, 896, 7, 7), (43904, 49, 7, 1), 0); del buf485  # reuse
        buf1534 = reinterpret_tensor(buf1596, (8, 112, 7, 7), (43904, 49, 7, 1), 0)  # alias
        buf1842 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_379, sp_380], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1526, buf1530, buf1531, primals_368, primals_369, buf1534, buf1842, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_369
        # Source Nodes: [sp_382], Original ATen: [aten.convolution]
        buf1535 = extern_kernels.convolution(reinterpret_tensor(buf1524, (8, 112, 14, 14), (175616, 1, 12544, 896), 112), buf93, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1535, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1536 = reinterpret_tensor(buf1525, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1525  # reuse
        # Source Nodes: [sp_382], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1535, buf1536, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1537 = buf1529; del buf1529  # reuse
        buf1538 = buf1528; del buf1528  # reuse
        buf1539 = buf1527; del buf1527  # reuse
        # Source Nodes: [sp_383], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1536, buf1537, buf1538, buf1539, 448, 98, grid=grid(448), stream=stream0)
        buf1540 = buf1531; del buf1531  # reuse
        buf1541 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1543 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_383], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1537, buf1538, buf1539, primals_819, primals_820, buf1540, buf1541, buf1543, primals_819, primals_820, 112, 4, grid=grid(112), stream=stream0)
        del primals_819
        del primals_820
        buf1544 = reinterpret_tensor(buf1596, (8, 112, 7, 7), (43904, 49, 7, 1), 5488)  # alias
        buf1841 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_383, sp_384], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1536, buf1540, buf1541, primals_371, primals_372, buf1544, buf1841, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_372
        # Source Nodes: [sp_386], Original ATen: [aten.convolution]
        buf1545 = extern_kernels.convolution(reinterpret_tensor(buf1524, (8, 112, 14, 14), (175616, 1, 12544, 896), 224), buf94, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1545, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1546 = reinterpret_tensor(buf1535, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1535  # reuse
        # Source Nodes: [sp_386], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1545, buf1546, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1547 = buf1539; del buf1539  # reuse
        buf1548 = buf1538; del buf1538  # reuse
        buf1549 = buf1537; del buf1537  # reuse
        # Source Nodes: [sp_387], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1546, buf1547, buf1548, buf1549, 448, 98, grid=grid(448), stream=stream0)
        buf1550 = buf1541; del buf1541  # reuse
        buf1551 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1553 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_387], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1547, buf1548, buf1549, primals_822, primals_823, buf1550, buf1551, buf1553, primals_822, primals_823, 112, 4, grid=grid(112), stream=stream0)
        del primals_822
        del primals_823
        buf1554 = reinterpret_tensor(buf1596, (8, 112, 7, 7), (43904, 49, 7, 1), 10976)  # alias
        buf1840 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_387, sp_388], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1546, buf1550, buf1551, primals_374, primals_375, buf1554, buf1840, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_375
        # Source Nodes: [sp_390], Original ATen: [aten.convolution]
        buf1555 = extern_kernels.convolution(reinterpret_tensor(buf1524, (8, 112, 14, 14), (175616, 1, 12544, 896), 336), buf95, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1555, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1556 = reinterpret_tensor(buf1545, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1545  # reuse
        # Source Nodes: [sp_390], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1555, buf1556, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1557 = buf1549; del buf1549  # reuse
        buf1558 = buf1548; del buf1548  # reuse
        buf1559 = buf1547; del buf1547  # reuse
        # Source Nodes: [sp_391], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1556, buf1557, buf1558, buf1559, 448, 98, grid=grid(448), stream=stream0)
        buf1560 = buf1551; del buf1551  # reuse
        buf1561 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1563 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_391], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1557, buf1558, buf1559, primals_825, primals_826, buf1560, buf1561, buf1563, primals_825, primals_826, 112, 4, grid=grid(112), stream=stream0)
        del primals_825
        del primals_826
        buf1564 = reinterpret_tensor(buf1596, (8, 112, 7, 7), (43904, 49, 7, 1), 16464)  # alias
        buf1839 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_391, sp_392], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1556, buf1560, buf1561, primals_377, primals_378, buf1564, buf1839, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_378
        # Source Nodes: [sp_394], Original ATen: [aten.convolution]
        buf1565 = extern_kernels.convolution(reinterpret_tensor(buf1524, (8, 112, 14, 14), (175616, 1, 12544, 896), 448), buf96, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1565, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1566 = reinterpret_tensor(buf1555, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1555  # reuse
        # Source Nodes: [sp_394], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1565, buf1566, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1567 = buf1559; del buf1559  # reuse
        buf1568 = buf1558; del buf1558  # reuse
        buf1569 = buf1557; del buf1557  # reuse
        # Source Nodes: [sp_395], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1566, buf1567, buf1568, buf1569, 448, 98, grid=grid(448), stream=stream0)
        buf1570 = buf1561; del buf1561  # reuse
        buf1571 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1573 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_395], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1567, buf1568, buf1569, primals_828, primals_829, buf1570, buf1571, buf1573, primals_828, primals_829, 112, 4, grid=grid(112), stream=stream0)
        del primals_828
        del primals_829
        buf1574 = reinterpret_tensor(buf1596, (8, 112, 7, 7), (43904, 49, 7, 1), 21952)  # alias
        buf1838 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_395, sp_396], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1566, buf1570, buf1571, primals_380, primals_381, buf1574, buf1838, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_381
        # Source Nodes: [sp_398], Original ATen: [aten.convolution]
        buf1575 = extern_kernels.convolution(reinterpret_tensor(buf1524, (8, 112, 14, 14), (175616, 1, 12544, 896), 560), buf97, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1575, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1576 = reinterpret_tensor(buf1565, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1565  # reuse
        # Source Nodes: [sp_398], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1575, buf1576, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1577 = buf1569; del buf1569  # reuse
        buf1578 = buf1568; del buf1568  # reuse
        buf1579 = buf1567; del buf1567  # reuse
        # Source Nodes: [sp_399], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1576, buf1577, buf1578, buf1579, 448, 98, grid=grid(448), stream=stream0)
        buf1580 = buf1571; del buf1571  # reuse
        buf1581 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1583 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_399], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1577, buf1578, buf1579, primals_831, primals_832, buf1580, buf1581, buf1583, primals_831, primals_832, 112, 4, grid=grid(112), stream=stream0)
        del primals_831
        del primals_832
        buf1584 = reinterpret_tensor(buf1596, (8, 112, 7, 7), (43904, 49, 7, 1), 27440)  # alias
        buf1837 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_399, sp_400], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1576, buf1580, buf1581, primals_383, primals_384, buf1584, buf1837, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_384
        # Source Nodes: [sp_402], Original ATen: [aten.convolution]
        buf1585 = extern_kernels.convolution(reinterpret_tensor(buf1524, (8, 112, 14, 14), (175616, 1, 12544, 896), 672), buf98, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1585, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1586 = reinterpret_tensor(buf1575, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1575  # reuse
        # Source Nodes: [sp_402], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1585, buf1586, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1587 = buf1579; del buf1579  # reuse
        buf1588 = buf1578; del buf1578  # reuse
        buf1589 = buf1577; del buf1577  # reuse
        # Source Nodes: [sp_403], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1586, buf1587, buf1588, buf1589, 448, 98, grid=grid(448), stream=stream0)
        buf1590 = buf1581; del buf1581  # reuse
        buf1591 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1593 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_403], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1587, buf1588, buf1589, primals_834, primals_835, buf1590, buf1591, buf1593, primals_834, primals_835, 112, 4, grid=grid(112), stream=stream0)
        del primals_834
        del primals_835
        buf1594 = reinterpret_tensor(buf1596, (8, 112, 7, 7), (43904, 49, 7, 1), 32928)  # alias
        buf1836 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_403, sp_404], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1586, buf1590, buf1591, primals_386, primals_387, buf1594, buf1836, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_387
        buf1595 = reinterpret_tensor(buf1596, (8, 112, 7, 7), (43904, 49, 7, 1), 38416)  # alias
        # Source Nodes: [getattr_l__mod___layer4___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_95.run(buf1524, buf1595, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1597 = empty_strided((8, 896, 7, 7), (43904, 1, 6272, 896), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_18], Original ATen: [aten.cat]
        triton_poi_fused_cat_96.run(buf1596, buf1597, 7168, 49, grid=grid(7168, 49), stream=stream0)
        del buf1534
        del buf1544
        del buf1554
        del buf1564
        del buf1574
        del buf1584
        del buf1594
        del buf1595
        # Source Nodes: [out_108], Original ATen: [aten.convolution]
        buf1598 = extern_kernels.convolution(buf1597, primals_388, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1598, (8, 2048, 7, 7), (100352, 49, 7, 1))
        buf1599 = empty_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_108], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_97.run(buf1598, buf1599, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf1600 = empty_strided((1, 2048, 1, 1, 4), (8192, 1, 8192, 8192, 2048), device='cuda', dtype=torch.float32)
        buf1601 = empty_strided((1, 2048, 1, 1, 4), (8192, 1, 8192, 8192, 2048), device='cuda', dtype=torch.float32)
        buf1602 = empty_strided((1, 2048, 1, 1, 4), (8192, 1, 8192, 8192, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_109], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_98.run(buf1599, buf1600, buf1601, buf1602, 8192, 98, grid=grid(8192), stream=stream0)
        buf1603 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf1604 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf1606 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_109], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_99.run(buf1600, buf1601, buf1602, primals_837, primals_838, buf1603, buf1604, buf1606, primals_837, primals_838, 2048, 4, grid=grid(2048), stream=stream0)
        del primals_837
        del primals_838
        # Source Nodes: [getattr_l__mod___layer4___0___downsample_0], Original ATen: [aten.convolution]
        buf1607 = extern_kernels.convolution(buf1514, primals_391, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1607, (8, 2048, 7, 7), (100352, 49, 7, 1))
        buf1608 = reinterpret_tensor(buf1598, (8, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf1598  # reuse
        # Source Nodes: [getattr_l__mod___layer4___0___downsample_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_97.run(buf1607, buf1608, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf1609 = buf1602; del buf1602  # reuse
        buf1610 = buf1601; del buf1601  # reuse
        buf1611 = buf1600; del buf1600  # reuse
        # Source Nodes: [shortcut_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_98.run(buf1608, buf1609, buf1610, buf1611, 8192, 98, grid=grid(8192), stream=stream0)
        buf1612 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf1613 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf1615 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_99.run(buf1609, buf1610, buf1611, primals_840, primals_841, buf1612, buf1613, buf1615, primals_840, primals_841, 2048, 4, grid=grid(2048), stream=stream0)
        del primals_840
        del primals_841
        buf1616 = reinterpret_tensor(buf1607, (8, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf1607  # reuse
        buf1617 = buf1616; del buf1616  # reuse
        # Source Nodes: [out_109, out_110, shortcut_17, shortcut_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_100.run(buf1617, buf1599, buf1603, buf1604, primals_389, primals_390, buf1608, buf1612, buf1613, primals_392, primals_393, 802816, grid=grid(802816), stream=stream0)
        del primals_390
        del primals_393
        # Source Nodes: [out_112], Original ATen: [aten.convolution]
        buf1618 = extern_kernels.convolution(buf1617, primals_394, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1618, (8, 896, 7, 7), (43904, 49, 7, 1))
        buf1619 = reinterpret_tensor(buf1596, (8, 896, 7, 7), (43904, 1, 6272, 896), 0); del buf1596  # reuse
        # Source Nodes: [out_112], Original ATen: [aten.convolution]
        triton_poi_fused_cat_96.run(buf1618, buf1619, 7168, 49, grid=grid(7168, 49), stream=stream0)
        buf1620 = empty_strided((1, 896, 1, 1, 4), (3584, 1, 3584, 3584, 896), device='cuda', dtype=torch.float32)
        buf1621 = empty_strided((1, 896, 1, 1, 4), (3584, 1, 3584, 3584, 896), device='cuda', dtype=torch.float32)
        buf1622 = empty_strided((1, 896, 1, 1, 4), (3584, 1, 3584, 3584, 896), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_113], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_101.run(buf1619, buf1620, buf1621, buf1622, 3584, 98, grid=grid(3584), stream=stream0)
        buf1623 = buf1521; del buf1521  # reuse
        buf1624 = empty_strided((1, 896, 1, 1), (896, 1, 896, 896), device='cuda', dtype=torch.float32)
        buf1626 = empty((896, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_113], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_102.run(buf1620, buf1621, buf1622, primals_843, primals_844, buf1623, buf1624, buf1626, primals_843, primals_844, 896, 4, grid=grid(896), stream=stream0)
        del primals_843
        del primals_844
        buf1627 = reinterpret_tensor(buf1618, (8, 896, 7, 7), (43904, 1, 6272, 896), 0); del buf1618  # reuse
        buf1835 = empty_strided((8, 896, 7, 7), (43904, 1, 6272, 896), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_113, out_114], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_103.run(buf1619, buf1623, buf1624, primals_395, primals_396, buf1627, buf1835, 351232, grid=grid(351232), stream=stream0)
        del primals_396
        # Source Nodes: [sp_407], Original ATen: [aten.convolution]
        buf1628 = extern_kernels.convolution(reinterpret_tensor(buf1627, (8, 112, 7, 7), (43904, 1, 6272, 896), 0), buf99, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1628, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1629 = reinterpret_tensor(buf1585, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1585  # reuse
        # Source Nodes: [sp_407], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1628, buf1629, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1630 = buf1589; del buf1589  # reuse
        buf1631 = buf1588; del buf1588  # reuse
        buf1632 = buf1587; del buf1587  # reuse
        # Source Nodes: [sp_408], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1629, buf1630, buf1631, buf1632, 448, 98, grid=grid(448), stream=stream0)
        buf1633 = buf1591; del buf1591  # reuse
        buf1634 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1636 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_408], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1630, buf1631, buf1632, primals_846, primals_847, buf1633, buf1634, buf1636, primals_846, primals_847, 112, 4, grid=grid(112), stream=stream0)
        del primals_846
        del primals_847
        buf1705 = empty((8, 896, 7, 7), device='cuda', dtype=torch.float32)
        buf1637 = reinterpret_tensor(buf1705, (8, 112, 7, 7), (43904, 49, 7, 1), 0)  # alias
        buf1834 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_408, sp_409], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1629, buf1633, buf1634, primals_398, primals_399, buf1637, buf1834, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_399
        buf1638 = reinterpret_tensor(buf1628, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1628  # reuse
        # Source Nodes: [sp_410], Original ATen: [aten.add]
        triton_poi_fused_add_104.run(buf1637, buf1627, buf1638, 392, 112, grid=grid(392, 112), stream=stream0)
        # Source Nodes: [sp_411], Original ATen: [aten.convolution]
        buf1639 = extern_kernels.convolution(buf1638, buf100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1639, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1640 = reinterpret_tensor(buf517, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf517  # reuse
        # Source Nodes: [sp_411], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1639, buf1640, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1641 = buf1632; del buf1632  # reuse
        buf1642 = buf1631; del buf1631  # reuse
        buf1643 = buf1630; del buf1630  # reuse
        # Source Nodes: [sp_412], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1640, buf1641, buf1642, buf1643, 448, 98, grid=grid(448), stream=stream0)
        buf1644 = buf1634; del buf1634  # reuse
        buf1645 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1647 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_412], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1641, buf1642, buf1643, primals_849, primals_850, buf1644, buf1645, buf1647, primals_849, primals_850, 112, 4, grid=grid(112), stream=stream0)
        del primals_849
        del primals_850
        buf1648 = reinterpret_tensor(buf1705, (8, 112, 7, 7), (43904, 49, 7, 1), 5488)  # alias
        buf1833 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_412, sp_413], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1640, buf1644, buf1645, primals_401, primals_402, buf1648, buf1833, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_402
        buf1649 = reinterpret_tensor(buf1639, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1639  # reuse
        # Source Nodes: [sp_414], Original ATen: [aten.add]
        triton_poi_fused_add_105.run(buf1648, buf1627, buf1649, 392, 112, grid=grid(392, 112), stream=stream0)
        # Source Nodes: [sp_415], Original ATen: [aten.convolution]
        buf1650 = extern_kernels.convolution(buf1649, buf101, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1650, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1651 = reinterpret_tensor(buf516, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf516  # reuse
        # Source Nodes: [sp_415], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1650, buf1651, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1652 = buf1643; del buf1643  # reuse
        buf1653 = buf1642; del buf1642  # reuse
        buf1654 = buf1641; del buf1641  # reuse
        # Source Nodes: [sp_416], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1651, buf1652, buf1653, buf1654, 448, 98, grid=grid(448), stream=stream0)
        buf1655 = buf1645; del buf1645  # reuse
        buf1656 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1658 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_416], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1652, buf1653, buf1654, primals_852, primals_853, buf1655, buf1656, buf1658, primals_852, primals_853, 112, 4, grid=grid(112), stream=stream0)
        del primals_852
        del primals_853
        buf1659 = reinterpret_tensor(buf1705, (8, 112, 7, 7), (43904, 49, 7, 1), 10976)  # alias
        buf1832 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_416, sp_417], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1651, buf1655, buf1656, primals_404, primals_405, buf1659, buf1832, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_405
        buf1660 = reinterpret_tensor(buf1650, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1650  # reuse
        # Source Nodes: [sp_418], Original ATen: [aten.add]
        triton_poi_fused_add_106.run(buf1659, buf1627, buf1660, 392, 112, grid=grid(392, 112), stream=stream0)
        # Source Nodes: [sp_419], Original ATen: [aten.convolution]
        buf1661 = extern_kernels.convolution(buf1660, buf102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1661, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1662 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_419], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1661, buf1662, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1663 = buf1654; del buf1654  # reuse
        buf1664 = buf1653; del buf1653  # reuse
        buf1665 = buf1652; del buf1652  # reuse
        # Source Nodes: [sp_420], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1662, buf1663, buf1664, buf1665, 448, 98, grid=grid(448), stream=stream0)
        buf1666 = buf1656; del buf1656  # reuse
        buf1667 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1669 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_420], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1663, buf1664, buf1665, primals_855, primals_856, buf1666, buf1667, buf1669, primals_855, primals_856, 112, 4, grid=grid(112), stream=stream0)
        del primals_855
        del primals_856
        buf1670 = reinterpret_tensor(buf1705, (8, 112, 7, 7), (43904, 49, 7, 1), 16464)  # alias
        buf1831 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_420, sp_421], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1662, buf1666, buf1667, primals_407, primals_408, buf1670, buf1831, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_408
        buf1671 = reinterpret_tensor(buf1661, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1661  # reuse
        # Source Nodes: [sp_422], Original ATen: [aten.add]
        triton_poi_fused_add_107.run(buf1670, buf1627, buf1671, 392, 112, grid=grid(392, 112), stream=stream0)
        # Source Nodes: [sp_423], Original ATen: [aten.convolution]
        buf1672 = extern_kernels.convolution(buf1671, buf103, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1672, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1673 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_423], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1672, buf1673, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1674 = buf1665; del buf1665  # reuse
        buf1675 = buf1664; del buf1664  # reuse
        buf1676 = buf1663; del buf1663  # reuse
        # Source Nodes: [sp_424], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1673, buf1674, buf1675, buf1676, 448, 98, grid=grid(448), stream=stream0)
        buf1677 = buf1667; del buf1667  # reuse
        buf1678 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1680 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_424], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1674, buf1675, buf1676, primals_858, primals_859, buf1677, buf1678, buf1680, primals_858, primals_859, 112, 4, grid=grid(112), stream=stream0)
        del primals_858
        del primals_859
        buf1681 = reinterpret_tensor(buf1705, (8, 112, 7, 7), (43904, 49, 7, 1), 21952)  # alias
        buf1830 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_424, sp_425], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1673, buf1677, buf1678, primals_410, primals_411, buf1681, buf1830, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_411
        buf1682 = reinterpret_tensor(buf1672, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1672  # reuse
        # Source Nodes: [sp_426], Original ATen: [aten.add]
        triton_poi_fused_add_108.run(buf1681, buf1627, buf1682, 392, 112, grid=grid(392, 112), stream=stream0)
        # Source Nodes: [sp_427], Original ATen: [aten.convolution]
        buf1683 = extern_kernels.convolution(buf1682, buf104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1683, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1684 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_427], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1683, buf1684, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1685 = buf1676; del buf1676  # reuse
        buf1686 = buf1675; del buf1675  # reuse
        buf1687 = buf1674; del buf1674  # reuse
        # Source Nodes: [sp_428], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1684, buf1685, buf1686, buf1687, 448, 98, grid=grid(448), stream=stream0)
        buf1688 = buf1678; del buf1678  # reuse
        buf1689 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1691 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_428], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1685, buf1686, buf1687, primals_861, primals_862, buf1688, buf1689, buf1691, primals_861, primals_862, 112, 4, grid=grid(112), stream=stream0)
        del primals_861
        del primals_862
        buf1692 = reinterpret_tensor(buf1705, (8, 112, 7, 7), (43904, 49, 7, 1), 27440)  # alias
        buf1829 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_428, sp_429], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1684, buf1688, buf1689, primals_413, primals_414, buf1692, buf1829, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_414
        buf1693 = reinterpret_tensor(buf1683, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1683  # reuse
        # Source Nodes: [sp_430], Original ATen: [aten.add]
        triton_poi_fused_add_109.run(buf1692, buf1627, buf1693, 392, 112, grid=grid(392, 112), stream=stream0)
        # Source Nodes: [sp_431], Original ATen: [aten.convolution]
        buf1694 = extern_kernels.convolution(buf1693, buf105, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1694, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1695 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_431], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1694, buf1695, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1696 = buf1687; del buf1687  # reuse
        buf1697 = buf1686; del buf1686  # reuse
        buf1698 = buf1685; del buf1685  # reuse
        # Source Nodes: [sp_432], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1695, buf1696, buf1697, buf1698, 448, 98, grid=grid(448), stream=stream0)
        buf1699 = buf1689; del buf1689  # reuse
        buf1700 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1702 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_432], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1696, buf1697, buf1698, primals_864, primals_865, buf1699, buf1700, buf1702, primals_864, primals_865, 112, 4, grid=grid(112), stream=stream0)
        del primals_864
        del primals_865
        buf1703 = reinterpret_tensor(buf1705, (8, 112, 7, 7), (43904, 49, 7, 1), 32928)  # alias
        buf1828 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_432, sp_433], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1695, buf1699, buf1700, primals_416, primals_417, buf1703, buf1828, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_417
        buf1704 = reinterpret_tensor(buf1705, (8, 112, 7, 7), (43904, 49, 7, 1), 38416)  # alias
        # Source Nodes: [cat_17], Original ATen: [aten.cat]
        triton_poi_fused_cat_110.run(buf1627, buf1704, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1706 = empty_strided((8, 896, 7, 7), (43904, 1, 6272, 896), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_17], Original ATen: [aten.cat]
        triton_poi_fused_cat_96.run(buf1705, buf1706, 7168, 49, grid=grid(7168, 49), stream=stream0)
        del buf1637
        del buf1648
        del buf1659
        del buf1670
        del buf1681
        del buf1692
        del buf1703
        del buf1704
        # Source Nodes: [out_116], Original ATen: [aten.convolution]
        buf1707 = extern_kernels.convolution(buf1706, primals_418, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1707, (8, 2048, 7, 7), (100352, 49, 7, 1))
        buf1708 = empty_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_116], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_97.run(buf1707, buf1708, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf1709 = buf1611; del buf1611  # reuse
        buf1710 = buf1610; del buf1610  # reuse
        buf1711 = buf1609; del buf1609  # reuse
        # Source Nodes: [out_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_98.run(buf1708, buf1709, buf1710, buf1711, 8192, 98, grid=grid(8192), stream=stream0)
        buf1712 = buf1613; del buf1613  # reuse
        buf1713 = buf1604; del buf1604  # reuse
        buf1715 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_99.run(buf1709, buf1710, buf1711, primals_867, primals_868, buf1712, buf1713, buf1715, primals_867, primals_868, 2048, 4, grid=grid(2048), stream=stream0)
        del primals_867
        del primals_868
        buf1716 = reinterpret_tensor(buf1707, (8, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf1707  # reuse
        # Source Nodes: [out_117, out_118, shortcut_19], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_111.run(buf1708, buf1712, buf1713, primals_419, primals_420, buf1617, buf1716, 802816, grid=grid(802816), stream=stream0)
        del primals_420
        # Source Nodes: [out_120], Original ATen: [aten.convolution]
        buf1717 = extern_kernels.convolution(buf1716, primals_421, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1717, (8, 896, 7, 7), (43904, 49, 7, 1))
        buf1718 = reinterpret_tensor(buf1705, (8, 896, 7, 7), (43904, 1, 6272, 896), 0); del buf1705  # reuse
        # Source Nodes: [out_120], Original ATen: [aten.convolution]
        triton_poi_fused_cat_96.run(buf1717, buf1718, 7168, 49, grid=grid(7168, 49), stream=stream0)
        buf1719 = buf1622; del buf1622  # reuse
        buf1720 = buf1621; del buf1621  # reuse
        buf1721 = buf1620; del buf1620  # reuse
        # Source Nodes: [out_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_101.run(buf1718, buf1719, buf1720, buf1721, 3584, 98, grid=grid(3584), stream=stream0)
        buf1722 = buf1624; del buf1624  # reuse
        buf1723 = empty_strided((1, 896, 1, 1), (896, 1, 896, 896), device='cuda', dtype=torch.float32)
        buf1725 = empty((896, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_102.run(buf1719, buf1720, buf1721, primals_870, primals_871, buf1722, buf1723, buf1725, primals_870, primals_871, 896, 4, grid=grid(896), stream=stream0)
        del buf1719
        del buf1720
        del buf1721
        del primals_870
        del primals_871
        buf1726 = reinterpret_tensor(buf1717, (8, 896, 7, 7), (43904, 1, 6272, 896), 0); del buf1717  # reuse
        buf1827 = empty_strided((8, 896, 7, 7), (43904, 1, 6272, 896), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_121, out_122], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_103.run(buf1718, buf1722, buf1723, primals_422, primals_423, buf1726, buf1827, 351232, grid=grid(351232), stream=stream0)
        del buf1723
        del primals_423
        # Source Nodes: [sp_436], Original ATen: [aten.convolution]
        buf1727 = extern_kernels.convolution(reinterpret_tensor(buf1726, (8, 112, 7, 7), (43904, 1, 6272, 896), 0), buf106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1727, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1728 = reinterpret_tensor(buf1694, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1694  # reuse
        # Source Nodes: [sp_436], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1727, buf1728, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1729 = buf1698; del buf1698  # reuse
        buf1730 = buf1697; del buf1697  # reuse
        buf1731 = buf1696; del buf1696  # reuse
        # Source Nodes: [sp_437], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1728, buf1729, buf1730, buf1731, 448, 98, grid=grid(448), stream=stream0)
        buf1732 = buf1700; del buf1700  # reuse
        buf1733 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1735 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_437], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1729, buf1730, buf1731, primals_873, primals_874, buf1732, buf1733, buf1735, primals_873, primals_874, 112, 4, grid=grid(112), stream=stream0)
        del primals_873
        del primals_874
        buf1804 = empty((8, 896, 7, 7), device='cuda', dtype=torch.float32)
        buf1736 = reinterpret_tensor(buf1804, (8, 112, 7, 7), (43904, 49, 7, 1), 0)  # alias
        buf1826 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_437, sp_438], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1728, buf1732, buf1733, primals_425, primals_426, buf1736, buf1826, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_426
        buf1737 = reinterpret_tensor(buf1727, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1727  # reuse
        # Source Nodes: [sp_439], Original ATen: [aten.add]
        triton_poi_fused_add_104.run(buf1736, buf1726, buf1737, 392, 112, grid=grid(392, 112), stream=stream0)
        # Source Nodes: [sp_440], Original ATen: [aten.convolution]
        buf1738 = extern_kernels.convolution(buf1737, buf107, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1738, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1739 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_440], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1738, buf1739, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1740 = buf1731; del buf1731  # reuse
        buf1741 = buf1730; del buf1730  # reuse
        buf1742 = buf1729; del buf1729  # reuse
        # Source Nodes: [sp_441], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1739, buf1740, buf1741, buf1742, 448, 98, grid=grid(448), stream=stream0)
        buf1743 = buf1733; del buf1733  # reuse
        buf1744 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1746 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_441], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1740, buf1741, buf1742, primals_876, primals_877, buf1743, buf1744, buf1746, primals_876, primals_877, 112, 4, grid=grid(112), stream=stream0)
        del primals_876
        del primals_877
        buf1747 = reinterpret_tensor(buf1804, (8, 112, 7, 7), (43904, 49, 7, 1), 5488)  # alias
        buf1825 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_441, sp_442], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1739, buf1743, buf1744, primals_428, primals_429, buf1747, buf1825, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_429
        buf1748 = reinterpret_tensor(buf1738, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1738  # reuse
        # Source Nodes: [sp_443], Original ATen: [aten.add]
        triton_poi_fused_add_105.run(buf1747, buf1726, buf1748, 392, 112, grid=grid(392, 112), stream=stream0)
        # Source Nodes: [sp_444], Original ATen: [aten.convolution]
        buf1749 = extern_kernels.convolution(buf1748, buf108, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1749, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1750 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_444], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1749, buf1750, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1751 = buf1742; del buf1742  # reuse
        buf1752 = buf1741; del buf1741  # reuse
        buf1753 = buf1740; del buf1740  # reuse
        # Source Nodes: [sp_445], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1750, buf1751, buf1752, buf1753, 448, 98, grid=grid(448), stream=stream0)
        buf1754 = buf1744; del buf1744  # reuse
        buf1755 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1757 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_445], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1751, buf1752, buf1753, primals_879, primals_880, buf1754, buf1755, buf1757, primals_879, primals_880, 112, 4, grid=grid(112), stream=stream0)
        del primals_879
        del primals_880
        buf1758 = reinterpret_tensor(buf1804, (8, 112, 7, 7), (43904, 49, 7, 1), 10976)  # alias
        buf1824 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_445, sp_446], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1750, buf1754, buf1755, primals_431, primals_432, buf1758, buf1824, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_432
        buf1759 = reinterpret_tensor(buf1749, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1749  # reuse
        # Source Nodes: [sp_447], Original ATen: [aten.add]
        triton_poi_fused_add_106.run(buf1758, buf1726, buf1759, 392, 112, grid=grid(392, 112), stream=stream0)
        # Source Nodes: [sp_448], Original ATen: [aten.convolution]
        buf1760 = extern_kernels.convolution(buf1759, buf109, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1760, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1761 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_448], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1760, buf1761, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1762 = buf1753; del buf1753  # reuse
        buf1763 = buf1752; del buf1752  # reuse
        buf1764 = buf1751; del buf1751  # reuse
        # Source Nodes: [sp_449], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1761, buf1762, buf1763, buf1764, 448, 98, grid=grid(448), stream=stream0)
        buf1765 = buf1755; del buf1755  # reuse
        buf1766 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1768 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_449], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1762, buf1763, buf1764, primals_882, primals_883, buf1765, buf1766, buf1768, primals_882, primals_883, 112, 4, grid=grid(112), stream=stream0)
        del primals_882
        del primals_883
        buf1769 = reinterpret_tensor(buf1804, (8, 112, 7, 7), (43904, 49, 7, 1), 16464)  # alias
        buf1823 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_449, sp_450], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1761, buf1765, buf1766, primals_434, primals_435, buf1769, buf1823, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_435
        buf1770 = reinterpret_tensor(buf1760, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1760  # reuse
        # Source Nodes: [sp_451], Original ATen: [aten.add]
        triton_poi_fused_add_107.run(buf1769, buf1726, buf1770, 392, 112, grid=grid(392, 112), stream=stream0)
        # Source Nodes: [sp_452], Original ATen: [aten.convolution]
        buf1771 = extern_kernels.convolution(buf1770, buf110, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1771, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1772 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_452], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1771, buf1772, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1773 = buf1764; del buf1764  # reuse
        buf1774 = buf1763; del buf1763  # reuse
        buf1775 = buf1762; del buf1762  # reuse
        # Source Nodes: [sp_453], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1772, buf1773, buf1774, buf1775, 448, 98, grid=grid(448), stream=stream0)
        buf1776 = buf1766; del buf1766  # reuse
        buf1777 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1779 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_453], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1773, buf1774, buf1775, primals_885, primals_886, buf1776, buf1777, buf1779, primals_885, primals_886, 112, 4, grid=grid(112), stream=stream0)
        del primals_885
        del primals_886
        buf1780 = reinterpret_tensor(buf1804, (8, 112, 7, 7), (43904, 49, 7, 1), 21952)  # alias
        buf1822 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_453, sp_454], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1772, buf1776, buf1777, primals_437, primals_438, buf1780, buf1822, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_438
        buf1781 = reinterpret_tensor(buf1771, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1771  # reuse
        # Source Nodes: [sp_455], Original ATen: [aten.add]
        triton_poi_fused_add_108.run(buf1780, buf1726, buf1781, 392, 112, grid=grid(392, 112), stream=stream0)
        # Source Nodes: [sp_456], Original ATen: [aten.convolution]
        buf1782 = extern_kernels.convolution(buf1781, buf111, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1782, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1783 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_456], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1782, buf1783, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1784 = buf1775; del buf1775  # reuse
        buf1785 = buf1774; del buf1774  # reuse
        buf1786 = buf1773; del buf1773  # reuse
        # Source Nodes: [sp_457], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1783, buf1784, buf1785, buf1786, 448, 98, grid=grid(448), stream=stream0)
        buf1787 = buf1777; del buf1777  # reuse
        buf1788 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1790 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_457], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1784, buf1785, buf1786, primals_888, primals_889, buf1787, buf1788, buf1790, primals_888, primals_889, 112, 4, grid=grid(112), stream=stream0)
        del primals_888
        del primals_889
        buf1791 = reinterpret_tensor(buf1804, (8, 112, 7, 7), (43904, 49, 7, 1), 27440)  # alias
        buf1821 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_457, sp_458], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1783, buf1787, buf1788, primals_440, primals_441, buf1791, buf1821, 896, 49, grid=grid(896, 49), stream=stream0)
        del primals_441
        buf1792 = reinterpret_tensor(buf1782, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf1782  # reuse
        # Source Nodes: [sp_459], Original ATen: [aten.add]
        triton_poi_fused_add_109.run(buf1791, buf1726, buf1792, 392, 112, grid=grid(392, 112), stream=stream0)
        # Source Nodes: [sp_460], Original ATen: [aten.convolution]
        buf1793 = extern_kernels.convolution(buf1792, buf112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1793, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf1794 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_460], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf1793, buf1794, 896, 49, grid=grid(896, 49), stream=stream0)
        del buf1793
        buf1795 = buf1786; del buf1786  # reuse
        buf1796 = buf1785; del buf1785  # reuse
        buf1797 = buf1784; del buf1784  # reuse
        # Source Nodes: [sp_461], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf1794, buf1795, buf1796, buf1797, 448, 98, grid=grid(448), stream=stream0)
        buf1798 = buf1788; del buf1788  # reuse
        buf1799 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf1801 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_461], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf1795, buf1796, buf1797, primals_891, primals_892, buf1798, buf1799, buf1801, primals_891, primals_892, 112, 4, grid=grid(112), stream=stream0)
        del buf1795
        del buf1796
        del buf1797
        del primals_891
        del primals_892
        buf1802 = reinterpret_tensor(buf1804, (8, 112, 7, 7), (43904, 49, 7, 1), 32928)  # alias
        buf1820 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_461, sp_462], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf1794, buf1798, buf1799, primals_443, primals_444, buf1802, buf1820, 896, 49, grid=grid(896, 49), stream=stream0)
        del buf1799
        del primals_444
        buf1803 = reinterpret_tensor(buf1804, (8, 112, 7, 7), (43904, 49, 7, 1), 38416)  # alias
        # Source Nodes: [cat_16], Original ATen: [aten.cat]
        triton_poi_fused_cat_110.run(buf1726, buf1803, 896, 49, grid=grid(896, 49), stream=stream0)
        buf1805 = empty_strided((8, 896, 7, 7), (43904, 1, 6272, 896), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_16], Original ATen: [aten.cat]
        triton_poi_fused_cat_96.run(buf1804, buf1805, 7168, 49, grid=grid(7168, 49), stream=stream0)
        del buf1736
        del buf1747
        del buf1758
        del buf1769
        del buf1780
        del buf1791
        del buf1802
        del buf1803
        del buf1804
        # Source Nodes: [out_124], Original ATen: [aten.convolution]
        buf1806 = extern_kernels.convolution(buf1805, primals_445, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1806, (8, 2048, 7, 7), (100352, 49, 7, 1))
        buf1807 = empty_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_124], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_97.run(buf1806, buf1807, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf1808 = buf1711; del buf1711  # reuse
        buf1809 = buf1710; del buf1710  # reuse
        buf1810 = buf1709; del buf1709  # reuse
        # Source Nodes: [out_125], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_98.run(buf1807, buf1808, buf1809, buf1810, 8192, 98, grid=grid(8192), stream=stream0)
        buf1811 = buf1713; del buf1713  # reuse
        buf1812 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf1814 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_125], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_99.run(buf1808, buf1809, buf1810, primals_894, primals_895, buf1811, buf1812, buf1814, primals_894, primals_895, 2048, 4, grid=grid(2048), stream=stream0)
        del buf1808
        del buf1809
        del buf1810
        del primals_894
        del primals_895
        buf1815 = reinterpret_tensor(buf1806, (8, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf1806  # reuse
        buf1819 = empty_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_125, out_126, x_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_112.run(buf1807, buf1811, buf1812, primals_446, primals_447, buf1716, buf1815, buf1819, 802816, grid=grid(802816), stream=stream0)
        del buf1812
        del primals_447
        buf1816 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cuda', dtype=torch.float32)
        buf1817 = reinterpret_tensor(buf1816, (8, 2048), (2048, 1), 0); del buf1816  # reuse
        # Source Nodes: [x_11, x_9], Original ATen: [aten.mean, aten.view]
        triton_per_fused_mean_view_113.run(buf1817, buf1815, 16384, 49, grid=grid(16384), stream=stream0)
        del buf1815
        buf1818 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_449, buf1817, reinterpret_tensor(primals_448, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf1818)
        del primals_449
        # Source Nodes: [x_1], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_452, primals_452, 1, grid=grid(1), stream=stream0)
        del primals_452
        # Source Nodes: [out_1], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_455, primals_455, 1, grid=grid(1), stream=stream0)
        del primals_455
        # Source Nodes: [sp_2], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_458, primals_458, 1, grid=grid(1), stream=stream0)
        del primals_458
        # Source Nodes: [sp_6], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_461, primals_461, 1, grid=grid(1), stream=stream0)
        del primals_461
        # Source Nodes: [sp_10], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_464, primals_464, 1, grid=grid(1), stream=stream0)
        del primals_464
        # Source Nodes: [sp_14], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_467, primals_467, 1, grid=grid(1), stream=stream0)
        del primals_467
        # Source Nodes: [sp_18], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_470, primals_470, 1, grid=grid(1), stream=stream0)
        del primals_470
        # Source Nodes: [sp_22], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_473, primals_473, 1, grid=grid(1), stream=stream0)
        del primals_473
        # Source Nodes: [sp_26], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_476, primals_476, 1, grid=grid(1), stream=stream0)
        del primals_476
        # Source Nodes: [out_5], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_479, primals_479, 1, grid=grid(1), stream=stream0)
        del primals_479
        # Source Nodes: [shortcut_1], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_482, primals_482, 1, grid=grid(1), stream=stream0)
        del primals_482
        # Source Nodes: [out_9], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_485, primals_485, 1, grid=grid(1), stream=stream0)
        del primals_485
        # Source Nodes: [sp_31], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_488, primals_488, 1, grid=grid(1), stream=stream0)
        del primals_488
        # Source Nodes: [sp_35], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_491, primals_491, 1, grid=grid(1), stream=stream0)
        del primals_491
        # Source Nodes: [sp_39], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_494, primals_494, 1, grid=grid(1), stream=stream0)
        del primals_494
        # Source Nodes: [sp_43], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_497, primals_497, 1, grid=grid(1), stream=stream0)
        del primals_497
        # Source Nodes: [sp_47], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_500, primals_500, 1, grid=grid(1), stream=stream0)
        del primals_500
        # Source Nodes: [sp_51], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_503, primals_503, 1, grid=grid(1), stream=stream0)
        del primals_503
        # Source Nodes: [sp_55], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_506, primals_506, 1, grid=grid(1), stream=stream0)
        del primals_506
        # Source Nodes: [out_13], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_509, primals_509, 1, grid=grid(1), stream=stream0)
        del primals_509
        # Source Nodes: [out_17], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_512, primals_512, 1, grid=grid(1), stream=stream0)
        del primals_512
        # Source Nodes: [sp_60], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_515, primals_515, 1, grid=grid(1), stream=stream0)
        del primals_515
        # Source Nodes: [sp_64], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_518, primals_518, 1, grid=grid(1), stream=stream0)
        del primals_518
        # Source Nodes: [sp_68], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_521, primals_521, 1, grid=grid(1), stream=stream0)
        del primals_521
        # Source Nodes: [sp_72], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_524, primals_524, 1, grid=grid(1), stream=stream0)
        del primals_524
        # Source Nodes: [sp_76], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_527, primals_527, 1, grid=grid(1), stream=stream0)
        del primals_527
        # Source Nodes: [sp_80], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_530, primals_530, 1, grid=grid(1), stream=stream0)
        del primals_530
        # Source Nodes: [sp_84], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_533, primals_533, 1, grid=grid(1), stream=stream0)
        del primals_533
        # Source Nodes: [out_21], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_536, primals_536, 1, grid=grid(1), stream=stream0)
        del primals_536
        # Source Nodes: [out_25], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_539, primals_539, 1, grid=grid(1), stream=stream0)
        del primals_539
        # Source Nodes: [sp_89], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_542, primals_542, 1, grid=grid(1), stream=stream0)
        del primals_542
        # Source Nodes: [sp_93], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_545, primals_545, 1, grid=grid(1), stream=stream0)
        del primals_545
        # Source Nodes: [sp_97], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_548, primals_548, 1, grid=grid(1), stream=stream0)
        del primals_548
        # Source Nodes: [sp_101], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_551, primals_551, 1, grid=grid(1), stream=stream0)
        del primals_551
        # Source Nodes: [sp_105], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_554, primals_554, 1, grid=grid(1), stream=stream0)
        del primals_554
        # Source Nodes: [sp_109], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_557, primals_557, 1, grid=grid(1), stream=stream0)
        del primals_557
        # Source Nodes: [sp_113], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_560, primals_560, 1, grid=grid(1), stream=stream0)
        del primals_560
        # Source Nodes: [out_29], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_563, primals_563, 1, grid=grid(1), stream=stream0)
        del primals_563
        # Source Nodes: [shortcut_5], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_566, primals_566, 1, grid=grid(1), stream=stream0)
        del primals_566
        # Source Nodes: [out_33], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_569, primals_569, 1, grid=grid(1), stream=stream0)
        del primals_569
        # Source Nodes: [sp_118], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_572, primals_572, 1, grid=grid(1), stream=stream0)
        del primals_572
        # Source Nodes: [sp_122], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_575, primals_575, 1, grid=grid(1), stream=stream0)
        del primals_575
        # Source Nodes: [sp_126], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_578, primals_578, 1, grid=grid(1), stream=stream0)
        del primals_578
        # Source Nodes: [sp_130], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_581, primals_581, 1, grid=grid(1), stream=stream0)
        del primals_581
        # Source Nodes: [sp_134], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_584, primals_584, 1, grid=grid(1), stream=stream0)
        del primals_584
        # Source Nodes: [sp_138], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_587, primals_587, 1, grid=grid(1), stream=stream0)
        del primals_587
        # Source Nodes: [sp_142], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_590, primals_590, 1, grid=grid(1), stream=stream0)
        del primals_590
        # Source Nodes: [out_37], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_593, primals_593, 1, grid=grid(1), stream=stream0)
        del primals_593
        # Source Nodes: [out_41], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_596, primals_596, 1, grid=grid(1), stream=stream0)
        del primals_596
        # Source Nodes: [sp_147], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_599, primals_599, 1, grid=grid(1), stream=stream0)
        del primals_599
        # Source Nodes: [sp_151], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_602, primals_602, 1, grid=grid(1), stream=stream0)
        del primals_602
        # Source Nodes: [sp_155], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_605, primals_605, 1, grid=grid(1), stream=stream0)
        del primals_605
        # Source Nodes: [sp_159], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_608, primals_608, 1, grid=grid(1), stream=stream0)
        del primals_608
        # Source Nodes: [sp_163], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_611, primals_611, 1, grid=grid(1), stream=stream0)
        del primals_611
        # Source Nodes: [sp_167], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_614, primals_614, 1, grid=grid(1), stream=stream0)
        del primals_614
        # Source Nodes: [sp_171], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_617, primals_617, 1, grid=grid(1), stream=stream0)
        del primals_617
        # Source Nodes: [out_45], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_620, primals_620, 1, grid=grid(1), stream=stream0)
        del primals_620
        # Source Nodes: [out_49], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_623, primals_623, 1, grid=grid(1), stream=stream0)
        del primals_623
        # Source Nodes: [sp_176], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_626, primals_626, 1, grid=grid(1), stream=stream0)
        del primals_626
        # Source Nodes: [sp_180], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_629, primals_629, 1, grid=grid(1), stream=stream0)
        del primals_629
        # Source Nodes: [sp_184], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_632, primals_632, 1, grid=grid(1), stream=stream0)
        del primals_632
        # Source Nodes: [sp_188], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_635, primals_635, 1, grid=grid(1), stream=stream0)
        del primals_635
        # Source Nodes: [sp_192], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_638, primals_638, 1, grid=grid(1), stream=stream0)
        del primals_638
        # Source Nodes: [sp_196], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_641, primals_641, 1, grid=grid(1), stream=stream0)
        del primals_641
        # Source Nodes: [sp_200], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_644, primals_644, 1, grid=grid(1), stream=stream0)
        del primals_644
        # Source Nodes: [out_53], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_647, primals_647, 1, grid=grid(1), stream=stream0)
        del primals_647
        # Source Nodes: [out_57], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_650, primals_650, 1, grid=grid(1), stream=stream0)
        del primals_650
        # Source Nodes: [sp_205], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_653, primals_653, 1, grid=grid(1), stream=stream0)
        del primals_653
        # Source Nodes: [sp_209], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_656, primals_656, 1, grid=grid(1), stream=stream0)
        del primals_656
        # Source Nodes: [sp_213], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_659, primals_659, 1, grid=grid(1), stream=stream0)
        del primals_659
        # Source Nodes: [sp_217], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_662, primals_662, 1, grid=grid(1), stream=stream0)
        del primals_662
        # Source Nodes: [sp_221], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_665, primals_665, 1, grid=grid(1), stream=stream0)
        del primals_665
        # Source Nodes: [sp_225], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_668, primals_668, 1, grid=grid(1), stream=stream0)
        del primals_668
        # Source Nodes: [sp_229], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_671, primals_671, 1, grid=grid(1), stream=stream0)
        del primals_671
        # Source Nodes: [out_61], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_674, primals_674, 1, grid=grid(1), stream=stream0)
        del primals_674
        # Source Nodes: [shortcut_10], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_677, primals_677, 1, grid=grid(1), stream=stream0)
        del primals_677
        # Source Nodes: [out_65], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_680, primals_680, 1, grid=grid(1), stream=stream0)
        del primals_680
        # Source Nodes: [sp_234], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_683, primals_683, 1, grid=grid(1), stream=stream0)
        del primals_683
        # Source Nodes: [sp_238], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_686, primals_686, 1, grid=grid(1), stream=stream0)
        del primals_686
        # Source Nodes: [sp_242], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_689, primals_689, 1, grid=grid(1), stream=stream0)
        del primals_689
        # Source Nodes: [sp_246], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_692, primals_692, 1, grid=grid(1), stream=stream0)
        del primals_692
        # Source Nodes: [sp_250], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_695, primals_695, 1, grid=grid(1), stream=stream0)
        del primals_695
        # Source Nodes: [sp_254], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_698, primals_698, 1, grid=grid(1), stream=stream0)
        del primals_698
        # Source Nodes: [sp_258], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_701, primals_701, 1, grid=grid(1), stream=stream0)
        del primals_701
        # Source Nodes: [out_69], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_704, primals_704, 1, grid=grid(1), stream=stream0)
        del primals_704
        # Source Nodes: [out_73], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_707, primals_707, 1, grid=grid(1), stream=stream0)
        del primals_707
        # Source Nodes: [sp_263], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_710, primals_710, 1, grid=grid(1), stream=stream0)
        del primals_710
        # Source Nodes: [sp_267], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_713, primals_713, 1, grid=grid(1), stream=stream0)
        del primals_713
        # Source Nodes: [sp_271], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_716, primals_716, 1, grid=grid(1), stream=stream0)
        del primals_716
        # Source Nodes: [sp_275], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_719, primals_719, 1, grid=grid(1), stream=stream0)
        del primals_719
        # Source Nodes: [sp_279], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_722, primals_722, 1, grid=grid(1), stream=stream0)
        del primals_722
        # Source Nodes: [sp_283], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_725, primals_725, 1, grid=grid(1), stream=stream0)
        del primals_725
        # Source Nodes: [sp_287], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_728, primals_728, 1, grid=grid(1), stream=stream0)
        del primals_728
        # Source Nodes: [out_77], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_731, primals_731, 1, grid=grid(1), stream=stream0)
        del primals_731
        # Source Nodes: [out_81], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_734, primals_734, 1, grid=grid(1), stream=stream0)
        del primals_734
        # Source Nodes: [sp_292], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_737, primals_737, 1, grid=grid(1), stream=stream0)
        del primals_737
        # Source Nodes: [sp_296], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_740, primals_740, 1, grid=grid(1), stream=stream0)
        del primals_740
        # Source Nodes: [sp_300], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_743, primals_743, 1, grid=grid(1), stream=stream0)
        del primals_743
        # Source Nodes: [sp_304], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_746, primals_746, 1, grid=grid(1), stream=stream0)
        del primals_746
        # Source Nodes: [sp_308], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_749, primals_749, 1, grid=grid(1), stream=stream0)
        del primals_749
        # Source Nodes: [sp_312], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_752, primals_752, 1, grid=grid(1), stream=stream0)
        del primals_752
        # Source Nodes: [sp_316], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_755, primals_755, 1, grid=grid(1), stream=stream0)
        del primals_755
        # Source Nodes: [out_85], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_758, primals_758, 1, grid=grid(1), stream=stream0)
        del primals_758
        # Source Nodes: [out_89], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_761, primals_761, 1, grid=grid(1), stream=stream0)
        del primals_761
        # Source Nodes: [sp_321], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_764, primals_764, 1, grid=grid(1), stream=stream0)
        del primals_764
        # Source Nodes: [sp_325], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_767, primals_767, 1, grid=grid(1), stream=stream0)
        del primals_767
        # Source Nodes: [sp_329], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_770, primals_770, 1, grid=grid(1), stream=stream0)
        del primals_770
        # Source Nodes: [sp_333], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_773, primals_773, 1, grid=grid(1), stream=stream0)
        del primals_773
        # Source Nodes: [sp_337], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_776, primals_776, 1, grid=grid(1), stream=stream0)
        del primals_776
        # Source Nodes: [sp_341], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_779, primals_779, 1, grid=grid(1), stream=stream0)
        del primals_779
        # Source Nodes: [sp_345], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_782, primals_782, 1, grid=grid(1), stream=stream0)
        del primals_782
        # Source Nodes: [out_93], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_785, primals_785, 1, grid=grid(1), stream=stream0)
        del primals_785
        # Source Nodes: [out_97], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_788, primals_788, 1, grid=grid(1), stream=stream0)
        del primals_788
        # Source Nodes: [sp_350], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_791, primals_791, 1, grid=grid(1), stream=stream0)
        del primals_791
        # Source Nodes: [sp_354], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_794, primals_794, 1, grid=grid(1), stream=stream0)
        del primals_794
        # Source Nodes: [sp_358], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_797, primals_797, 1, grid=grid(1), stream=stream0)
        del primals_797
        # Source Nodes: [sp_362], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_800, primals_800, 1, grid=grid(1), stream=stream0)
        del primals_800
        # Source Nodes: [sp_366], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_803, primals_803, 1, grid=grid(1), stream=stream0)
        del primals_803
        # Source Nodes: [sp_370], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_806, primals_806, 1, grid=grid(1), stream=stream0)
        del primals_806
        # Source Nodes: [sp_374], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_809, primals_809, 1, grid=grid(1), stream=stream0)
        del primals_809
        # Source Nodes: [out_101], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_812, primals_812, 1, grid=grid(1), stream=stream0)
        del primals_812
        # Source Nodes: [out_105], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_815, primals_815, 1, grid=grid(1), stream=stream0)
        del primals_815
        # Source Nodes: [sp_379], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_818, primals_818, 1, grid=grid(1), stream=stream0)
        del primals_818
        # Source Nodes: [sp_383], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_821, primals_821, 1, grid=grid(1), stream=stream0)
        del primals_821
        # Source Nodes: [sp_387], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_824, primals_824, 1, grid=grid(1), stream=stream0)
        del primals_824
        # Source Nodes: [sp_391], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_827, primals_827, 1, grid=grid(1), stream=stream0)
        del primals_827
        # Source Nodes: [sp_395], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_830, primals_830, 1, grid=grid(1), stream=stream0)
        del primals_830
        # Source Nodes: [sp_399], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_833, primals_833, 1, grid=grid(1), stream=stream0)
        del primals_833
        # Source Nodes: [sp_403], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_836, primals_836, 1, grid=grid(1), stream=stream0)
        del primals_836
        # Source Nodes: [out_109], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_839, primals_839, 1, grid=grid(1), stream=stream0)
        del primals_839
        # Source Nodes: [shortcut_17], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_842, primals_842, 1, grid=grid(1), stream=stream0)
        del primals_842
        # Source Nodes: [out_113], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_845, primals_845, 1, grid=grid(1), stream=stream0)
        del primals_845
        # Source Nodes: [sp_408], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_848, primals_848, 1, grid=grid(1), stream=stream0)
        del primals_848
        # Source Nodes: [sp_412], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_851, primals_851, 1, grid=grid(1), stream=stream0)
        del primals_851
        # Source Nodes: [sp_416], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_854, primals_854, 1, grid=grid(1), stream=stream0)
        del primals_854
        # Source Nodes: [sp_420], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_857, primals_857, 1, grid=grid(1), stream=stream0)
        del primals_857
        # Source Nodes: [sp_424], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_860, primals_860, 1, grid=grid(1), stream=stream0)
        del primals_860
        # Source Nodes: [sp_428], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_863, primals_863, 1, grid=grid(1), stream=stream0)
        del primals_863
        # Source Nodes: [sp_432], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_866, primals_866, 1, grid=grid(1), stream=stream0)
        del primals_866
        # Source Nodes: [out_117], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_869, primals_869, 1, grid=grid(1), stream=stream0)
        del primals_869
        # Source Nodes: [out_121], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_872, primals_872, 1, grid=grid(1), stream=stream0)
        del primals_872
        # Source Nodes: [sp_437], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_875, primals_875, 1, grid=grid(1), stream=stream0)
        del primals_875
        # Source Nodes: [sp_441], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_878, primals_878, 1, grid=grid(1), stream=stream0)
        del primals_878
        # Source Nodes: [sp_445], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_881, primals_881, 1, grid=grid(1), stream=stream0)
        del primals_881
        # Source Nodes: [sp_449], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_884, primals_884, 1, grid=grid(1), stream=stream0)
        del primals_884
        # Source Nodes: [sp_453], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_887, primals_887, 1, grid=grid(1), stream=stream0)
        del primals_887
        # Source Nodes: [sp_457], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_890, primals_890, 1, grid=grid(1), stream=stream0)
        del primals_890
        # Source Nodes: [sp_461], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_893, primals_893, 1, grid=grid(1), stream=stream0)
        del primals_893
        # Source Nodes: [out_125], Original ATen: [aten.add]
        triton_poi_fused_add_114.run(primals_896, primals_896, 1, grid=grid(1), stream=stream0)
        del primals_896
        return (buf1818, buf0, primals_2, primals_4, primals_5, buf1, primals_8, buf2, primals_11, buf3, primals_14, buf4, primals_17, buf5, primals_20, buf6, primals_23, buf7, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, buf8, primals_38, buf9, primals_41, buf10, primals_44, buf11, primals_47, buf12, primals_50, buf13, primals_53, buf14, primals_56, primals_58, primals_59, primals_61, primals_62, buf15, primals_65, buf16, primals_68, buf17, primals_71, buf18, primals_74, buf19, primals_77, buf20, primals_80, buf21, primals_83, primals_85, primals_86, primals_88, primals_89, buf22, primals_92, buf23, primals_95, buf24, primals_98, buf25, primals_101, buf26, primals_104, buf27, primals_107, buf28, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, buf29, primals_122, buf30, primals_125, buf31, primals_128, buf32, primals_131, buf33, primals_134, buf34, primals_137, buf35, primals_140, primals_142, primals_143, primals_145, primals_146, buf36, primals_149, buf37, primals_152, buf38, primals_155, buf39, primals_158, buf40, primals_161, buf41, primals_164, buf42, primals_167, primals_169, primals_170, primals_172, primals_173, buf43, primals_176, buf44, primals_179, buf45, primals_182, buf46, primals_185, buf47, primals_188, buf48, primals_191, buf49, primals_194, primals_196, primals_197, primals_199, primals_200, buf50, primals_203, buf51, primals_206, buf52, primals_209, buf53, primals_212, buf54, primals_215, buf55, primals_218, buf56, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, buf57, primals_233, buf58, primals_236, buf59, primals_239, buf60, primals_242, buf61, primals_245, buf62, primals_248, buf63, primals_251, primals_253, primals_254, primals_256, primals_257, buf64, primals_260, buf65, primals_263, buf66, primals_266, buf67, primals_269, buf68, primals_272, buf69, primals_275, buf70, primals_278, primals_280, primals_281, primals_283, primals_284, buf71, primals_287, buf72, primals_290, buf73, primals_293, buf74, primals_296, buf75, primals_299, buf76, primals_302, buf77, primals_305, primals_307, primals_308, primals_310, primals_311, buf78, primals_314, buf79, primals_317, buf80, primals_320, buf81, primals_323, buf82, primals_326, buf83, primals_329, buf84, primals_332, primals_334, primals_335, primals_337, primals_338, buf85, primals_341, buf86, primals_344, buf87, primals_347, buf88, primals_350, buf89, primals_353, buf90, primals_356, buf91, primals_359, primals_361, primals_362, primals_364, primals_365, buf92, primals_368, buf93, primals_371, buf94, primals_374, buf95, primals_377, buf96, primals_380, buf97, primals_383, buf98, primals_386, primals_388, primals_389, primals_391, primals_392, primals_394, primals_395, buf99, primals_398, buf100, primals_401, buf101, primals_404, buf102, primals_407, buf103, primals_410, buf104, primals_413, buf105, primals_416, primals_418, primals_419, primals_421, primals_422, buf106, primals_425, buf107, primals_428, buf108, primals_431, buf109, primals_434, buf110, primals_437, buf111, primals_440, buf112, primals_443, primals_445, primals_446, buf113, buf115, buf125, buf126, buf127, buf128, buf130, buf140, reinterpret_tensor(buf141, (8, 14, 56, 56), (351232, 1, 6272, 112), 0), buf143, buf153, reinterpret_tensor(buf141, (8, 14, 56, 56), (351232, 1, 6272, 112), 14), buf156, buf166, reinterpret_tensor(buf141, (8, 14, 56, 56), (351232, 1, 6272, 112), 28), buf169, buf179, reinterpret_tensor(buf141, (8, 14, 56, 56), (351232, 1, 6272, 112), 42), buf182, buf192, reinterpret_tensor(buf141, (8, 14, 56, 56), (351232, 1, 6272, 112), 56), buf195, buf205, reinterpret_tensor(buf141, (8, 14, 56, 56), (351232, 1, 6272, 112), 70), buf208, buf218, reinterpret_tensor(buf141, (8, 14, 56, 56), (351232, 1, 6272, 112), 84), buf221, buf231, reinterpret_tensor(buf141, (8, 14, 56, 56), (351232, 1, 6272, 112), 98), buf235, buf237, buf247, buf249, buf259, buf261, buf263, buf273, reinterpret_tensor(buf274, (8, 14, 56, 56), (351232, 1, 6272, 112), 0), buf276, buf286, buf288, buf290, buf300, buf302, buf304, buf314, buf316, buf318, buf328, buf330, buf332, buf342, buf344, buf346, buf356, buf358, buf360, buf370, buf374, buf376, buf386, buf387, buf389, buf399, reinterpret_tensor(buf400, (8, 14, 56, 56), (351232, 1, 6272, 112), 0), buf402, buf412, buf414, buf416, buf426, buf428, buf430, buf440, buf442, buf444, buf454, buf456, buf458, buf468, buf470, buf472, buf482, buf484, buf486, buf496, buf500, buf502, buf512, buf513, buf515, buf525, reinterpret_tensor(buf526, (8, 28, 56, 56), (702464, 1, 12544, 224), 0), buf528, buf535, reinterpret_tensor(buf526, (8, 28, 56, 56), (702464, 1, 12544, 224), 28), buf538, buf545, reinterpret_tensor(buf526, (8, 28, 56, 56), (702464, 1, 12544, 224), 56), buf548, buf555, reinterpret_tensor(buf526, (8, 28, 56, 56), (702464, 1, 12544, 224), 84), buf558, buf565, reinterpret_tensor(buf526, (8, 28, 56, 56), (702464, 1, 12544, 224), 112), buf568, buf575, reinterpret_tensor(buf526, (8, 28, 56, 56), (702464, 1, 12544, 224), 140), buf578, buf585, reinterpret_tensor(buf526, (8, 28, 56, 56), (702464, 1, 12544, 224), 168), buf588, buf595, reinterpret_tensor(buf526, (8, 28, 56, 56), (702464, 1, 12544, 224), 196), buf599, buf601, buf608, buf610, buf617, buf619, buf621, buf628, reinterpret_tensor(buf629, (8, 28, 28, 28), (175616, 1, 6272, 224), 0), buf631, buf638, buf640, buf642, buf649, buf651, buf653, buf660, buf662, buf664, buf671, buf673, buf675, buf682, buf684, buf686, buf693, buf695, buf697, buf704, buf708, buf710, buf717, buf718, buf720, buf727, reinterpret_tensor(buf728, (8, 28, 28, 28), (175616, 1, 6272, 224), 0), buf730, buf737, buf739, buf741, buf748, buf750, buf752, buf759, buf761, buf763, buf770, buf772, buf774, buf781, buf783, buf785, buf792, buf794, buf796, buf803, buf807, buf809, buf816, buf817, buf819, buf826, reinterpret_tensor(buf827, (8, 28, 28, 28), (175616, 1, 6272, 224), 0), buf829, buf836, buf838, buf840, buf847, buf849, buf851, buf858, buf860, buf862, buf869, buf871, buf873, buf880, buf882, buf884, buf891, buf893, buf895, buf902, buf906, buf908, buf915, buf916, buf918, buf925, reinterpret_tensor(buf926, (8, 56, 28, 28), (351232, 1, 12544, 448), 0), buf928, buf935, reinterpret_tensor(buf926, (8, 56, 28, 28), (351232, 1, 12544, 448), 56), buf938, buf945, reinterpret_tensor(buf926, (8, 56, 28, 28), (351232, 1, 12544, 448), 112), buf948, buf955, reinterpret_tensor(buf926, (8, 56, 28, 28), (351232, 1, 12544, 448), 168), buf958, buf965, reinterpret_tensor(buf926, (8, 56, 28, 28), (351232, 1, 12544, 448), 224), buf968, buf975, reinterpret_tensor(buf926, (8, 56, 28, 28), (351232, 1, 12544, 448), 280), buf978, buf985, reinterpret_tensor(buf926, (8, 56, 28, 28), (351232, 1, 12544, 448), 336), buf988, buf995, reinterpret_tensor(buf926, (8, 56, 28, 28), (351232, 1, 12544, 448), 392), buf999, buf1001, buf1008, buf1010, buf1017, buf1019, buf1021, buf1028, reinterpret_tensor(buf1029, (8, 56, 14, 14), (87808, 1, 6272, 448), 0), buf1031, buf1038, buf1040, buf1042, buf1049, buf1051, buf1053, buf1060, buf1062, buf1064, buf1071, buf1073, buf1075, buf1082, buf1084, buf1086, buf1093, buf1095, buf1097, buf1104, buf1108, buf1110, buf1117, buf1118, buf1120, buf1127, reinterpret_tensor(buf1128, (8, 56, 14, 14), (87808, 1, 6272, 448), 0), buf1130, buf1137, buf1139, buf1141, buf1148, buf1150, buf1152, buf1159, buf1161, buf1163, buf1170, buf1172, buf1174, buf1181, buf1183, buf1185, buf1192, buf1194, buf1196, buf1203, buf1207, buf1209, buf1216, buf1217, buf1219, buf1226, reinterpret_tensor(buf1227, (8, 56, 14, 14), (87808, 1, 6272, 448), 0), buf1229, buf1236, buf1238, buf1240, buf1247, buf1249, buf1251, buf1258, buf1260, buf1262, buf1269, buf1271, buf1273, buf1280, buf1282, buf1284, buf1291, buf1293, buf1295, buf1302, buf1306, buf1308, buf1315, buf1316, buf1318, buf1325, reinterpret_tensor(buf1326, (8, 56, 14, 14), (87808, 1, 6272, 448), 0), buf1328, buf1335, buf1337, buf1339, buf1346, buf1348, buf1350, buf1357, buf1359, buf1361, buf1368, buf1370, buf1372, buf1379, buf1381, buf1383, buf1390, buf1392, buf1394, buf1401, buf1405, buf1407, buf1414, buf1415, buf1417, buf1424, reinterpret_tensor(buf1425, (8, 56, 14, 14), (87808, 1, 6272, 448), 0), buf1427, buf1434, buf1436, buf1438, buf1445, buf1447, buf1449, buf1456, buf1458, buf1460, buf1467, buf1469, buf1471, buf1478, buf1480, buf1482, buf1489, buf1491, buf1493, buf1500, buf1504, buf1506, buf1513, buf1514, buf1516, buf1523, reinterpret_tensor(buf1524, (8, 112, 14, 14), (175616, 1, 12544, 896), 0), buf1526, buf1533, reinterpret_tensor(buf1524, (8, 112, 14, 14), (175616, 1, 12544, 896), 112), buf1536, buf1543, reinterpret_tensor(buf1524, (8, 112, 14, 14), (175616, 1, 12544, 896), 224), buf1546, buf1553, reinterpret_tensor(buf1524, (8, 112, 14, 14), (175616, 1, 12544, 896), 336), buf1556, buf1563, reinterpret_tensor(buf1524, (8, 112, 14, 14), (175616, 1, 12544, 896), 448), buf1566, buf1573, reinterpret_tensor(buf1524, (8, 112, 14, 14), (175616, 1, 12544, 896), 560), buf1576, buf1583, reinterpret_tensor(buf1524, (8, 112, 14, 14), (175616, 1, 12544, 896), 672), buf1586, buf1593, reinterpret_tensor(buf1524, (8, 112, 14, 14), (175616, 1, 12544, 896), 784), buf1597, buf1599, buf1606, buf1608, buf1615, buf1617, buf1619, buf1626, reinterpret_tensor(buf1627, (8, 112, 7, 7), (43904, 1, 6272, 896), 0), buf1629, buf1636, buf1638, buf1640, buf1647, buf1649, buf1651, buf1658, buf1660, buf1662, buf1669, buf1671, buf1673, buf1680, buf1682, buf1684, buf1691, buf1693, buf1695, buf1702, buf1706, buf1708, buf1715, buf1716, buf1718, buf1725, reinterpret_tensor(buf1726, (8, 112, 7, 7), (43904, 1, 6272, 896), 0), buf1728, buf1735, buf1737, buf1739, buf1746, buf1748, buf1750, buf1757, buf1759, buf1761, buf1768, buf1770, buf1772, buf1779, buf1781, buf1783, buf1790, buf1792, buf1794, buf1801, buf1805, buf1807, buf1814, buf1817, reinterpret_tensor(primals_448, (1000, 2048), (2048, 1), 0), buf1819, reinterpret_tensor(buf1811, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), buf1820, reinterpret_tensor(buf1798, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1821, reinterpret_tensor(buf1787, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1822, reinterpret_tensor(buf1776, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1823, reinterpret_tensor(buf1765, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1824, reinterpret_tensor(buf1754, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1825, reinterpret_tensor(buf1743, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1826, reinterpret_tensor(buf1732, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1827, reinterpret_tensor(buf1722, (1, 896, 1, 1), (896, 1, 1, 1), 0), reinterpret_tensor(buf1712, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), buf1828, reinterpret_tensor(buf1699, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1829, reinterpret_tensor(buf1688, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1830, reinterpret_tensor(buf1677, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1831, reinterpret_tensor(buf1666, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1832, reinterpret_tensor(buf1655, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1833, reinterpret_tensor(buf1644, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1834, reinterpret_tensor(buf1633, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1835, reinterpret_tensor(buf1623, (1, 896, 1, 1), (896, 1, 1, 1), 0), reinterpret_tensor(buf1612, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf1603, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), buf1836, reinterpret_tensor(buf1590, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1837, reinterpret_tensor(buf1580, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1838, reinterpret_tensor(buf1570, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1839, reinterpret_tensor(buf1560, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1840, reinterpret_tensor(buf1550, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1841, reinterpret_tensor(buf1540, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1842, reinterpret_tensor(buf1530, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf1843, reinterpret_tensor(buf1520, (1, 896, 1, 1), (896, 1, 1, 1), 0), reinterpret_tensor(buf1510, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf1844, reinterpret_tensor(buf1497, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1845, reinterpret_tensor(buf1486, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1846, reinterpret_tensor(buf1475, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1847, reinterpret_tensor(buf1464, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1848, reinterpret_tensor(buf1453, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1849, reinterpret_tensor(buf1442, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1850, reinterpret_tensor(buf1431, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1851, reinterpret_tensor(buf1421, (1, 448, 1, 1), (448, 1, 1, 1), 0), reinterpret_tensor(buf1411, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf1852, reinterpret_tensor(buf1398, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1853, reinterpret_tensor(buf1387, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1854, reinterpret_tensor(buf1376, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1855, reinterpret_tensor(buf1365, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1856, reinterpret_tensor(buf1354, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1857, reinterpret_tensor(buf1343, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1858, reinterpret_tensor(buf1332, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1859, reinterpret_tensor(buf1322, (1, 448, 1, 1), (448, 1, 1, 1), 0), reinterpret_tensor(buf1312, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf1860, reinterpret_tensor(buf1299, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1861, reinterpret_tensor(buf1288, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1862, reinterpret_tensor(buf1277, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1863, reinterpret_tensor(buf1266, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1864, reinterpret_tensor(buf1255, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1865, reinterpret_tensor(buf1244, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1866, reinterpret_tensor(buf1233, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1867, reinterpret_tensor(buf1223, (1, 448, 1, 1), (448, 1, 1, 1), 0), reinterpret_tensor(buf1213, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf1868, reinterpret_tensor(buf1200, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1869, reinterpret_tensor(buf1189, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1870, reinterpret_tensor(buf1178, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1871, reinterpret_tensor(buf1167, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1872, reinterpret_tensor(buf1156, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1873, reinterpret_tensor(buf1145, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1874, reinterpret_tensor(buf1134, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1875, reinterpret_tensor(buf1124, (1, 448, 1, 1), (448, 1, 1, 1), 0), reinterpret_tensor(buf1114, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf1876, reinterpret_tensor(buf1101, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1877, reinterpret_tensor(buf1090, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1878, reinterpret_tensor(buf1079, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1879, reinterpret_tensor(buf1068, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1880, reinterpret_tensor(buf1057, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1881, reinterpret_tensor(buf1046, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1882, reinterpret_tensor(buf1035, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1883, reinterpret_tensor(buf1025, (1, 448, 1, 1), (448, 1, 1, 1), 0), reinterpret_tensor(buf1014, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf1005, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf1884, reinterpret_tensor(buf992, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1885, reinterpret_tensor(buf982, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1886, reinterpret_tensor(buf972, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1887, reinterpret_tensor(buf962, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1888, reinterpret_tensor(buf952, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1889, reinterpret_tensor(buf942, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1890, reinterpret_tensor(buf932, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf1891, reinterpret_tensor(buf922, (1, 448, 1, 1), (448, 1, 1, 1), 0), reinterpret_tensor(buf912, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf1892, reinterpret_tensor(buf899, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1893, reinterpret_tensor(buf888, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1894, reinterpret_tensor(buf877, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1895, reinterpret_tensor(buf866, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1896, reinterpret_tensor(buf855, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1897, reinterpret_tensor(buf844, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1898, reinterpret_tensor(buf833, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1899, reinterpret_tensor(buf823, (1, 224, 1, 1), (224, 1, 1, 1), 0), reinterpret_tensor(buf813, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf1900, reinterpret_tensor(buf800, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1901, reinterpret_tensor(buf789, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1902, reinterpret_tensor(buf778, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1903, reinterpret_tensor(buf767, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1904, reinterpret_tensor(buf756, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1905, reinterpret_tensor(buf745, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1906, reinterpret_tensor(buf734, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1907, reinterpret_tensor(buf724, (1, 224, 1, 1), (224, 1, 1, 1), 0), reinterpret_tensor(buf714, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf1908, reinterpret_tensor(buf701, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1909, reinterpret_tensor(buf690, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1910, reinterpret_tensor(buf679, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1911, reinterpret_tensor(buf668, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1912, reinterpret_tensor(buf657, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1913, reinterpret_tensor(buf646, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1914, reinterpret_tensor(buf635, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1915, reinterpret_tensor(buf625, (1, 224, 1, 1), (224, 1, 1, 1), 0), reinterpret_tensor(buf614, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf605, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf1916, reinterpret_tensor(buf592, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1917, reinterpret_tensor(buf582, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1918, reinterpret_tensor(buf572, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1919, reinterpret_tensor(buf562, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1920, reinterpret_tensor(buf552, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1921, reinterpret_tensor(buf542, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1922, reinterpret_tensor(buf532, (1, 28, 1, 1), (28, 1, 1, 1), 0), buf1923, reinterpret_tensor(buf522, (1, 224, 1, 1), (224, 1, 1, 1), 0), reinterpret_tensor(buf509, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf1924, reinterpret_tensor(buf493, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1925, reinterpret_tensor(buf479, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1926, reinterpret_tensor(buf465, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1927, reinterpret_tensor(buf451, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1928, reinterpret_tensor(buf437, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1929, reinterpret_tensor(buf423, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1930, reinterpret_tensor(buf409, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1931, reinterpret_tensor(buf396, (1, 112, 1, 1), (112, 1, 1, 1), 0), reinterpret_tensor(buf383, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf1932, reinterpret_tensor(buf367, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1933, reinterpret_tensor(buf353, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1934, reinterpret_tensor(buf339, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1935, reinterpret_tensor(buf325, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1936, reinterpret_tensor(buf311, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1937, reinterpret_tensor(buf297, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1938, reinterpret_tensor(buf283, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1939, reinterpret_tensor(buf270, (1, 112, 1, 1), (112, 1, 1, 1), 0), reinterpret_tensor(buf256, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf244, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf1940, reinterpret_tensor(buf228, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1941, reinterpret_tensor(buf215, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1942, reinterpret_tensor(buf202, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1943, reinterpret_tensor(buf189, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1944, reinterpret_tensor(buf176, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1945, reinterpret_tensor(buf163, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1946, reinterpret_tensor(buf150, (1, 14, 1, 1), (14, 1, 1, 1), 0), buf1947, reinterpret_tensor(buf137, (1, 112, 1, 1), (112, 1, 1, 1), 0), reinterpret_tensor(buf122, (1, 64, 1, 1), (64, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((112, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((256, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((112, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((256, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((112, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((224, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((224, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((224, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((512, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((224, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((512, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((448, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((1024, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((448, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((1024, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((448, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1024, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((448, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((1024, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((448, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((1024, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((448, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((1024, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((896, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((2048, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((896, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((2048, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((896, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((2048, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_453 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_456 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_459 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_462 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_465 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_468 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_471 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_474 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_477 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_480 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_483 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_486 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_489 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_492 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_495 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_498 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_501 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_504 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_507 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_510 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_513 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_516 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_519 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_522 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_525 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_528 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_531 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_534 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_537 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_540 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_543 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_546 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_549 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_552 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_555 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_558 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_561 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_564 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_567 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_570 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_573 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_576 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_579 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_582 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_585 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_588 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_591 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_594 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_597 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_600 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_603 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_606 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_609 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_612 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_615 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_618 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_621 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_624 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_627 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_630 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_633 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_636 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_639 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_642 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_645 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_648 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_651 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_654 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_657 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_660 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_663 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_666 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_669 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_672 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_675 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_678 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_681 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_684 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_687 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_689 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_690 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_692 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_693 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_695 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_696 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_698 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_699 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_701 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_702 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_704 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_705 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_707 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_708 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_710 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_711 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_712 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_713 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_714 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_715 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_716 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_717 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_718 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_719 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_720 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_721 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_722 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_723 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_724 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_725 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_726 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_727 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_728 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_729 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_730 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_731 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_732 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_733 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_734 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_735 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_736 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_737 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_738 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_739 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_740 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_741 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_742 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_743 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_744 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_745 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_746 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_747 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_748 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_749 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_750 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_751 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_752 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_753 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_754 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_755 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_756 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_757 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_758 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_759 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_760 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_761 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_762 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_763 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_764 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_765 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_766 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_767 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_768 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_769 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_770 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_771 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_772 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_773 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_774 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_775 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_776 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_777 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_778 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_779 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_780 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_781 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_782 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_783 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_784 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_785 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_786 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_787 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_788 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_789 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_790 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_791 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_792 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_793 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_794 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_795 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_796 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_797 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_798 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_799 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_800 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_801 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_802 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_803 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_804 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_805 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_806 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_807 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_808 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_809 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_810 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_811 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_812 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_813 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_814 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_815 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_816 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_817 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_818 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_819 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_820 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_821 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_822 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_823 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_824 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_825 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_826 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_827 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_828 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_829 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_830 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_831 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_832 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_833 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_834 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_835 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_836 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_837 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_838 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_839 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_840 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_841 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_842 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_843 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_844 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_845 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_846 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_847 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_848 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_849 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_850 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_851 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_852 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_853 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_854 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_855 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_856 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_857 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_858 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_859 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_860 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_861 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_862 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_863 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_864 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_865 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_866 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_867 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_868 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_869 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_870 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_871 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_872 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_873 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_874 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_875 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_876 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_877 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_878 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_879 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_880 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_881 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_882 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_883 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_884 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_885 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_886 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_887 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_888 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_889 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_890 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_891 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_892 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_893 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_894 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_895 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_896 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_897 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('res2net50_14w_8s', benchmark_compiled_module)
