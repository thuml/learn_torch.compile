
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


# kernel path: /tmp/torchinductor_youkaichao/7d/c7df2wzqzj65kpbrw3kvwd46dfszadkdpcxhoozudelgek5bfog5.py
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
    size_hints=[128, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
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


# kernel path: /tmp/torchinductor_youkaichao/vx/cvxmqdfmwlds4jm7vpwvcnp5o2a5kdopuzougxslwg3b6jjiy4ew.py
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
    size_hints=[4096, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (288*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bp/cbpkewmlsvbjjjfmicfiatr4iolpoik5u7pvl6ybpmz7b2fyejmc.py
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
    size_hints=[16384, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (1152*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lc/clcstnw2t34pszh5xsgz3xftktwdlioqyqda7kcyiebbcv533ucb.py
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
    size_hints=[32768, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (1152*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/36/c36oiagljczxxm76j6yz72kzdoewp76jeazhnqqoyws5htz65rnr.py
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
    ynumel = 36864
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


# kernel path: /tmp/torchinductor_youkaichao/un/cuntmmnj4a5ak5n7h6gceympxad2w36bxfzaiwzm53ck26exrjik.py
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
    size_hints=[32768, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25600
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 160
    y1 = (yindex // 160)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (160*x2) + (1440*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/is/cisa24gwz76hav3cw7m4zpnqwj2nh7qxuzahxxpwc33rojfkn3iq.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 65536
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
    tmp0 = tl.load(in_ptr0 + (x2 + (65536*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (196608*y1)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ul/culugdncxmiauiykvx2oqt62qdi7xn46kryaoxvsrlkkakkmguda.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 16384
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16384*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (524288*y1)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/56/c56j3wpw5f67cfjdhugb7gajv6qanvx3gdat6qwnwwjbcpyf55eo.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (4096*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/vc/cvcxkws6ekgezadupumkrcrsnkfung3pa4kbxborqvup54sjvj5d.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 8
    x1 = (xindex // 8)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (32*r2) + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (32*r2) + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (32*r2) + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (32*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (32*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (32*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2y/c2ygfck5g3wjg5az5vznmfnk7eq6nbwzu2ivyrkjloswamlqg6mz.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => add_1, add_2, add_3, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze_1, var_mean
triton_per_fused__native_batch_norm_legit_functional_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_10', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (32*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 131072.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000076294527394
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


# kernel path: /tmp/torchinductor_youkaichao/m7/cm7ga6lwadzjlidczj53fva43ugjc65byi4cr2t7pycejbqtdo54.py
# Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# shortcut => relu
# x_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
triton_poi_fused__native_batch_norm_legit_functional_relu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 131072.0
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


# kernel path: /tmp/torchinductor_youkaichao/bn/cbnkdm6v22xpx6g7lkwkpkmhzchmc2uau7gdz3mfbby2oqwgr4hi.py
# Source Nodes: [x_6], Original ATen: [aten.convolution]
# x_6 => convolution_1
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
    ynumel = 1024
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (4096*y3)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (524288*y1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kz/ckzgit6hlk5vif54oah2hmxbpg7ldww2aanssll4bek5digpogau.py
# Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
# x_7 => var_mean_1
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
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/yh/cyhuunqvrenctr7ckpesvc5i6bytz4esalsj5guohc5uwjmtap2f.py
# Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
# x_7 => var_mean_1
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (128*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (128*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (128*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u3/cu36gpa3aferzmitj33ktzjlvyc2o4jwwsqfeit3nxe44bw75dyr.py
# Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
# x_7 => add_6, add_7, add_8, mul_10, mul_11, mul_12, mul_8, mul_9, rsqrt_1, squeeze_4, var_mean_1
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
    xnumel = 128
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 32768.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.000030518509476
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


# kernel path: /tmp/torchinductor_youkaichao/rj/crj7zcoicufzqtgl7pgp6izrsqv672ac5dt4wfvxwpj344yqqko2.py
# Source Nodes: [x_11, x_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_11 => relu_1
# x_7 => add_6, add_9, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
triton_poi_fused__native_batch_norm_legit_functional_relu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 32768.0
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


# kernel path: /tmp/torchinductor_youkaichao/qy/cqyiwrow5zwakn3dofaxa4mnzabay5pmkfjx5is6p6b5g3w2hxo7.py
# Source Nodes: [shortcut_1, x_13, x_21, x_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_1 => relu_2
# x_13 => add_11, add_14, mul_14, mul_20, rsqrt_2, sub_2, var_mean_2
# x_21 => add_16, add_19, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
# x_25 => add_20
triton_poi_fused__native_batch_norm_legit_functional_add_relu_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
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
    tmp4 = 32768.0
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


# kernel path: /tmp/torchinductor_youkaichao/76/c76bz7svzwij2w4vnzwz3pq3h5buhidc4izsoyqrsit4ap5ht2i5.py
# Source Nodes: [x_26], Original ATen: [aten.convolution]
# x_26 => convolution_4
triton_poi_fused_convolution_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (196608*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vg/cvgcgndtws3nfoizrbicblphlcu5ydze226pvpemafzwq43x5iw3.py
# Source Nodes: [x_27], Original ATen: [aten._native_batch_norm_legit_functional]
# x_27 => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (24576*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ws/cwsqtmbjmtrm2ylwsn3ylnqwmzlver2zi3wbbhw36zxya7yvmjlr.py
# Source Nodes: [x_27], Original ATen: [aten._native_batch_norm_legit_functional]
# x_27 => add_22, add_23, add_24, mul_29, mul_30, mul_31, mul_32, mul_33, rsqrt_4, squeeze_13, var_mean_4
triton_per_fused__native_batch_norm_legit_functional_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 64
    RBLOCK: tl.constexpr = 64
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
    tmp16 = 8192.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0001220852154804
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


# kernel path: /tmp/torchinductor_youkaichao/a3/ca3d23duonnruzjeuf5zjisgolaxvf5mbivovtvnwxzc5h7sp7mp.py
# Source Nodes: [x_27, x_31], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_27 => add_22, add_25, mul_28, mul_34, rsqrt_4, sub_4, var_mean_4
# x_31 => relu_3
triton_poi_fused__native_batch_norm_legit_functional_relu_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
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
    tmp4 = 8192.0
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


# kernel path: /tmp/torchinductor_youkaichao/tf/ctf344bttra3f52ecekgfx4uenriyk4wguqut5xcrle5qco4tkwr.py
# Source Nodes: [shortcut_2, x_33, x_41, x_45], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_2 => relu_4
# x_33 => add_27, add_30, mul_35, mul_41, rsqrt_5, sub_5, var_mean_5
# x_41 => add_32, add_35, mul_42, mul_48, rsqrt_6, sub_6, var_mean_6
# x_45 => add_36
triton_poi_fused__native_batch_norm_legit_functional_add_relu_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
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
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 8192.0
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


# kernel path: /tmp/torchinductor_youkaichao/fu/cfuvbiqq56nginhce7ijwzrl4i7edo52457lmtnuqgpbz4ielpjf.py
# Source Nodes: [shortcut_3, x_53, x_60], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_3 => relu_6
# x_53 => add_43, add_46, mul_56, mul_62, rsqrt_8, sub_8, var_mean_8
# x_60 => add_47
triton_poi_fused__native_batch_norm_legit_functional_add_relu_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
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
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 8192.0
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


# kernel path: /tmp/torchinductor_youkaichao/fx/cfxgurha46b4zwtnut4kwfsvdtkpqdnpyfxcyyyttu5tlcobifcr.py
# Source Nodes: [x_61], Original ATen: [aten.convolution]
# x_61 => convolution_9
triton_poi_fused_convolution_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 160
    y1 = (yindex // 160)
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (160*x2) + (163840*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6x/c6xu5sd6dparo5c3vgpdexhah6sw2nd342ycp7ykbslvpwai6v4m.py
# Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
# x_62 => var_mean_9
triton_red_fused__native_batch_norm_legit_functional_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10240
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 160
    x1 = (xindex // 160)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (160*r2) + (20480*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/kd/ckd6kobpznisemyltqdsixlzvdha3cqrxlsrwmru7auunxzmlmvj.py
# Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
# x_62 => add_49, add_50, add_51, mul_64, mul_65, mul_66, mul_67, mul_68, rsqrt_9, squeeze_28, var_mean_9
triton_per_fused__native_batch_norm_legit_functional_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_26', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (160*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (160*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (160*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 8192.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0001220852154804
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


# kernel path: /tmp/torchinductor_youkaichao/zm/czmplrlwos4xkyebi3l6fpfchlgsxxlakevm4e7rzyy4bils4ktp.py
# Source Nodes: [x_62, x_66], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_62 => add_49, add_52, mul_63, mul_69, rsqrt_9, sub_9, var_mean_9
# x_66 => relu_7
triton_poi_fused__native_batch_norm_legit_functional_relu_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 160
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 8192.0
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


# kernel path: /tmp/torchinductor_youkaichao/w3/cw3znftnhbdhwquj2ibd64tnhakmnnw3yu2g5lspeakzxyscgifc.py
# Source Nodes: [x_67], Original ATen: [aten.convolution]
# x_67 => convolution_10
triton_poi_fused_convolution_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 160
    y1 = (yindex // 160)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (160*x2) + (40960*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xx/cxxctevofzlaw6tr4a5xzbmzbkj6lbkyk4zee36bpm2drvln4vod.py
# Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
# x_68 => var_mean_10
triton_red_fused__native_batch_norm_legit_functional_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 160
    x1 = (xindex // 160)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (160*r2) + (20480*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/kq/ckqwza3umnqsavxq5vqo5krkxa7fw7avid6njjvwdxbsslsu64ae.py
# Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
# x_68 => add_54, add_55, add_56, mul_71, mul_72, mul_73, mul_74, mul_75, rsqrt_10, squeeze_31, var_mean_10
triton_per_fused__native_batch_norm_legit_functional_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_30', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (160*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (160*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (160*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 2048.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0004885197850513
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


# kernel path: /tmp/torchinductor_youkaichao/h6/ch6z6hfre6hzilpefegx7vgvozbpip3loagvzz6gyssglzqhhk76.py
# Source Nodes: [x_68, x_72], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_68 => add_54, add_57, mul_70, mul_76, rsqrt_10, sub_10, var_mean_10
# x_72 => relu_8
triton_poi_fused__native_batch_norm_legit_functional_relu_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 160
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
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


# kernel path: /tmp/torchinductor_youkaichao/li/clipkehtknw4k4vdskatjex232wqu3xhh4tabuh6fwsst5r7ekca.py
# Source Nodes: [x_75], Original ATen: [aten.convolution]
# x_75 => convolution_11
triton_poi_fused_convolution_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5120
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 640
    y1 = (yindex // 640)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (640*x2) + (163840*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4x/c4xacbcio3kaazvmtomfxg6cyqk4ux4zsxm3ssu6j6wdu4lo2gp7.py
# Source Nodes: [x_76], Original ATen: [aten._native_batch_norm_legit_functional]
# x_76 => var_mean_11
triton_red_fused__native_batch_norm_legit_functional_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10240
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 640
    x1 = (xindex // 640)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (640*r2) + (81920*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/nb/cnb55hucku3lgw2eihc7p4uamz5wr5qws3wobx2eej3ef6ervc6k.py
# Source Nodes: [x_76], Original ATen: [aten._native_batch_norm_legit_functional]
# x_76 => add_59, add_60, add_61, mul_78, mul_79, mul_80, mul_81, mul_82, rsqrt_11, squeeze_34, var_mean_11
triton_per_fused__native_batch_norm_legit_functional_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_34', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (640*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (640*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (640*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 2048.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0004885197850513
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


# kernel path: /tmp/torchinductor_youkaichao/pt/cpt3lgjlgyoxmtj7gh5sxkel2cdr3eluh33ejmbzpuya26l2gtjk.py
# Source Nodes: [shortcut_4, x_76, x_84, x_88], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_4 => relu_9
# x_76 => add_59, add_62, mul_77, mul_83, rsqrt_11, sub_11, var_mean_11
# x_84 => add_64, add_67, mul_84, mul_90, rsqrt_12, sub_12, var_mean_12
# x_88 => add_68
triton_poi_fused__native_batch_norm_legit_functional_add_relu_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_35', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 640
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
    tmp4 = 2048.0
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


# kernel path: /tmp/torchinductor_youkaichao/qe/cqevmntten5sarul7vpkourtdcov7kq5uampbqfnqtj2sypq3gx2.py
# Source Nodes: [shortcut_5, x_104, x_111], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_5 => relu_12
# x_104 => add_80, add_83, mul_105, mul_111, rsqrt_15, sub_15, var_mean_15
# x_111 => add_84
triton_poi_fused__native_batch_norm_legit_functional_add_relu_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 640
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
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


# kernel path: /tmp/torchinductor_youkaichao/id/cidawnl4g2ksfxcbo7y5v24ojb2vdja7owqkmpk2sji57ud7nec4.py
# Source Nodes: [x_204], Original ATen: [aten.convolution]
# x_204 => convolution_28
triton_poi_fused_convolution_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 15360
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1920
    y1 = (yindex // 1920)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1920*x2) + (491520*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fr/cfr3vlfpthappkc54grhkfahf3vc63d6ougxctuvhaltcr5hfvpu.py
# Source Nodes: [x_205], Original ATen: [aten._native_batch_norm_legit_functional]
# x_205 => var_mean_28
triton_red_fused__native_batch_norm_legit_functional_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 30720
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1920
    x1 = (xindex // 1920)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1920*r2) + (245760*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bt/cbtvlzfck4kx7xfdkdnq7wzbt6q764crnki7m4x3vtfe32rhqcn4.py
# Source Nodes: [x_205], Original ATen: [aten._native_batch_norm_legit_functional]
# x_205 => add_150, add_151, add_152, mul_197, mul_198, mul_199, mul_200, mul_201, rsqrt_28, squeeze_85, var_mean_28
triton_per_fused__native_batch_norm_legit_functional_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_39', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1920*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1920*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (1920*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 2048.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0004885197850513
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


# kernel path: /tmp/torchinductor_youkaichao/pw/cpwzw24jsby75aselv4wzpv7awy6cgfnlsla73rexabch4akrmf7.py
# Source Nodes: [x_205, x_209], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_205 => add_150, add_153, mul_196, mul_202, rsqrt_28, sub_28, var_mean_28
# x_209 => relu_25
triton_poi_fused__native_batch_norm_legit_functional_relu_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3932160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1920
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
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


# kernel path: /tmp/torchinductor_youkaichao/lm/clm5u7ghu5jcgit5hsaftifmrfzfytkxrmuj3ojp57zkhq32cdz2.py
# Source Nodes: [x_210], Original ATen: [aten.convolution]
# x_210 => convolution_29
triton_poi_fused_convolution_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 15360
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1920
    y1 = (yindex // 1920)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1920*x2) + (122880*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rw/crwnsae3mzb6jsfzai5o4fltfmjlp6aknt2cvdblbhkfyn4cpj6c.py
# Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_functional]
# x_211 => var_mean_29
triton_red_fused__native_batch_norm_legit_functional_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1920
    x1 = (xindex // 1920)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1920*r2) + (245760*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/at/catqd5dipkcbtuo43jcr5nxsxsquu7p4ybh27k7wx6nkp6spfwnr.py
# Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_functional]
# x_211 => add_155, add_156, add_157, mul_204, mul_205, mul_206, mul_207, mul_208, rsqrt_29, squeeze_88, var_mean_29
triton_per_fused__native_batch_norm_legit_functional_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_43', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1920*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1920*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (1920*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 512.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0019569471624266
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


# kernel path: /tmp/torchinductor_youkaichao/pc/cpcxpnyfbs3evdtha5tgri7jn2bycb57iwuuauhv4ivsvhllvsuk.py
# Source Nodes: [x_211, x_215], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_211 => add_155, add_158, mul_203, mul_209, rsqrt_29, sub_29, var_mean_29
# x_215 => relu_26
triton_poi_fused__native_batch_norm_legit_functional_relu_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 983040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1920
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
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


# kernel path: /tmp/torchinductor_youkaichao/4y/c4yrlhtpbsksyjrsl7smdleiyl2kujcz7hmaojzldheuyqodne5a.py
# Source Nodes: [x_218], Original ATen: [aten.convolution]
# x_218 => convolution_30
triton_poi_fused_convolution_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5120
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 640
    y1 = (yindex // 640)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (640*x2) + (40960*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6g/c6gycyf6pfaknvvgthcdieclqqgjoxomttaxrfg4tsohgu3bdugw.py
# Source Nodes: [x_219], Original ATen: [aten._native_batch_norm_legit_functional]
# x_219 => var_mean_30
triton_red_fused__native_batch_norm_legit_functional_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 640
    x1 = (xindex // 640)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (640*r2) + (81920*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/e4/ce42f5wcexv3xpa3p4e5ztt27wzsnhjuvdqkqvckb4ljk4mzfg7d.py
# Source Nodes: [x_219], Original ATen: [aten._native_batch_norm_legit_functional]
# x_219 => add_160, add_161, add_162, mul_211, mul_212, mul_213, mul_214, mul_215, rsqrt_30, squeeze_91, var_mean_30
triton_per_fused__native_batch_norm_legit_functional_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_47', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (640*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (640*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (640*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 512.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0019569471624266
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


# kernel path: /tmp/torchinductor_youkaichao/yp/cyp3faqye3sig36eoumryyptiv75r77aghxjqcye7rdyqzzxccme.py
# Source Nodes: [shortcut_10, x_219, x_227, x_231], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_10 => relu_27
# x_219 => add_160, add_163, mul_210, mul_216, rsqrt_30, sub_30, var_mean_30
# x_227 => add_165, add_168, mul_217, mul_223, rsqrt_31, sub_31, var_mean_31
# x_231 => add_169
triton_poi_fused__native_batch_norm_legit_functional_add_relu_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 640
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
    tmp4 = 512.0
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


# kernel path: /tmp/torchinductor_youkaichao/j6/cj6ev6wys2jxhrws7hnicmlmi3hwy2jfjyks6u4lw2dt5t3dkfjc.py
# Source Nodes: [shortcut_11, x_247, x_254], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_11 => relu_30
# x_247 => add_181, add_184, mul_238, mul_244, rsqrt_34, sub_34, var_mean_34
# x_254 => add_185
triton_poi_fused__native_batch_norm_legit_functional_add_relu_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 640
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
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


# kernel path: /tmp/torchinductor_youkaichao/n6/cn6a5c3zy62xctdkljmnv5jpofe6zyjekcxxgsmcp4hgvyxa2ewf.py
# Source Nodes: [x_417], Original ATen: [aten.convolution]
# x_417 => convolution_56
triton_poi_fused_convolution_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 20480
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 2560
    y1 = (yindex // 2560)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (2560*x2) + (163840*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6o/c6oa5kpcd44c7jte66w2in4ntjwl77k7g6epgrr5foi4brh5lt4w.py
# Source Nodes: [x_418], Original ATen: [aten._native_batch_norm_legit_functional]
# x_418 => var_mean_56
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
    xnumel = 10240
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2560
    x1 = (xindex // 2560)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2560*r2) + (327680*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4s/c4sygrgn2zwsgwmzx44xpafth5vf776xwbnfauoig3zczzylggbh.py
# Source Nodes: [x_418], Original ATen: [aten._native_batch_norm_legit_functional]
# x_418 => add_299, add_300, add_301, mul_393, mul_394, mul_395, mul_396, mul_397, rsqrt_56, squeeze_169, var_mean_56
triton_per_fused__native_batch_norm_legit_functional_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_52', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2560*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (2560*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (2560*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 512.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0019569471624266
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


# kernel path: /tmp/torchinductor_youkaichao/nm/cnm4sxftqnvn4b4pcq4fqkh2hdeluvtrjc533ovay7l4huosxy22.py
# Source Nodes: [x_418, x_423], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# x_418 => add_299, add_302, mul_392, mul_398, rsqrt_56, sub_56, var_mean_56
# x_423 => relu_52
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
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2560
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
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


# kernel path: /tmp/torchinductor_youkaichao/tu/ctuetx2qzmtmfte2h7aa77xoutxxr7fuevlze2q6sudphcrvwogf.py
# Source Nodes: [x_424, x_426], Original ATen: [aten.mean, aten.view]
# x_424 => mean
# x_426 => view
triton_per_fused_mean_view_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_view_54', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 2560
    x1 = (xindex // 2560)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2560*r2) + (163840*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 64.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ug/cugspe2x53e3iazwnd3lnhvlau6ixwdtpqedadohysisymeycnqc.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add
triton_poi_fused_add_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_55', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (192, ), (1, ))
    assert_size_stride(primals_10, (192, ), (1, ))
    assert_size_stride(primals_11, (192, ), (1, ))
    assert_size_stride(primals_12, (192, ), (1, ))
    assert_size_stride(primals_13, (192, ), (1, ))
    assert_size_stride(primals_14, (192, ), (1, ))
    assert_size_stride(primals_15, (192, ), (1, ))
    assert_size_stride(primals_16, (192, ), (1, ))
    assert_size_stride(primals_17, (192, ), (1, ))
    assert_size_stride(primals_18, (192, ), (1, ))
    assert_size_stride(primals_19, (160, ), (1, ))
    assert_size_stride(primals_20, (160, ), (1, ))
    assert_size_stride(primals_21, (160, ), (1, ))
    assert_size_stride(primals_22, (160, ), (1, ))
    assert_size_stride(primals_23, (640, ), (1, ))
    assert_size_stride(primals_24, (640, ), (1, ))
    assert_size_stride(primals_25, (640, ), (1, ))
    assert_size_stride(primals_26, (640, ), (1, ))
    assert_size_stride(primals_27, (160, ), (1, ))
    assert_size_stride(primals_28, (160, ), (1, ))
    assert_size_stride(primals_29, (160, ), (1, ))
    assert_size_stride(primals_30, (160, ), (1, ))
    assert_size_stride(primals_31, (640, ), (1, ))
    assert_size_stride(primals_32, (640, ), (1, ))
    assert_size_stride(primals_33, (160, ), (1, ))
    assert_size_stride(primals_34, (160, ), (1, ))
    assert_size_stride(primals_35, (160, ), (1, ))
    assert_size_stride(primals_36, (160, ), (1, ))
    assert_size_stride(primals_37, (640, ), (1, ))
    assert_size_stride(primals_38, (640, ), (1, ))
    assert_size_stride(primals_39, (160, ), (1, ))
    assert_size_stride(primals_40, (160, ), (1, ))
    assert_size_stride(primals_41, (160, ), (1, ))
    assert_size_stride(primals_42, (160, ), (1, ))
    assert_size_stride(primals_43, (640, ), (1, ))
    assert_size_stride(primals_44, (640, ), (1, ))
    assert_size_stride(primals_45, (160, ), (1, ))
    assert_size_stride(primals_46, (160, ), (1, ))
    assert_size_stride(primals_47, (160, ), (1, ))
    assert_size_stride(primals_48, (160, ), (1, ))
    assert_size_stride(primals_49, (640, ), (1, ))
    assert_size_stride(primals_50, (640, ), (1, ))
    assert_size_stride(primals_51, (160, ), (1, ))
    assert_size_stride(primals_52, (160, ), (1, ))
    assert_size_stride(primals_53, (160, ), (1, ))
    assert_size_stride(primals_54, (160, ), (1, ))
    assert_size_stride(primals_55, (640, ), (1, ))
    assert_size_stride(primals_56, (640, ), (1, ))
    assert_size_stride(primals_57, (1920, ), (1, ))
    assert_size_stride(primals_58, (1920, ), (1, ))
    assert_size_stride(primals_59, (1920, ), (1, ))
    assert_size_stride(primals_60, (1920, ), (1, ))
    assert_size_stride(primals_61, (640, ), (1, ))
    assert_size_stride(primals_62, (640, ), (1, ))
    assert_size_stride(primals_63, (640, ), (1, ))
    assert_size_stride(primals_64, (640, ), (1, ))
    assert_size_stride(primals_65, (1920, ), (1, ))
    assert_size_stride(primals_66, (1920, ), (1, ))
    assert_size_stride(primals_67, (1920, ), (1, ))
    assert_size_stride(primals_68, (1920, ), (1, ))
    assert_size_stride(primals_69, (640, ), (1, ))
    assert_size_stride(primals_70, (640, ), (1, ))
    assert_size_stride(primals_71, (1920, ), (1, ))
    assert_size_stride(primals_72, (1920, ), (1, ))
    assert_size_stride(primals_73, (1920, ), (1, ))
    assert_size_stride(primals_74, (1920, ), (1, ))
    assert_size_stride(primals_75, (640, ), (1, ))
    assert_size_stride(primals_76, (640, ), (1, ))
    assert_size_stride(primals_77, (1920, ), (1, ))
    assert_size_stride(primals_78, (1920, ), (1, ))
    assert_size_stride(primals_79, (1920, ), (1, ))
    assert_size_stride(primals_80, (1920, ), (1, ))
    assert_size_stride(primals_81, (640, ), (1, ))
    assert_size_stride(primals_82, (640, ), (1, ))
    assert_size_stride(primals_83, (1920, ), (1, ))
    assert_size_stride(primals_84, (1920, ), (1, ))
    assert_size_stride(primals_85, (1920, ), (1, ))
    assert_size_stride(primals_86, (1920, ), (1, ))
    assert_size_stride(primals_87, (640, ), (1, ))
    assert_size_stride(primals_88, (640, ), (1, ))
    assert_size_stride(primals_89, (1920, ), (1, ))
    assert_size_stride(primals_90, (1920, ), (1, ))
    assert_size_stride(primals_91, (1920, ), (1, ))
    assert_size_stride(primals_92, (1920, ), (1, ))
    assert_size_stride(primals_93, (640, ), (1, ))
    assert_size_stride(primals_94, (640, ), (1, ))
    assert_size_stride(primals_95, (1920, ), (1, ))
    assert_size_stride(primals_96, (1920, ), (1, ))
    assert_size_stride(primals_97, (1920, ), (1, ))
    assert_size_stride(primals_98, (1920, ), (1, ))
    assert_size_stride(primals_99, (640, ), (1, ))
    assert_size_stride(primals_100, (640, ), (1, ))
    assert_size_stride(primals_101, (1920, ), (1, ))
    assert_size_stride(primals_102, (1920, ), (1, ))
    assert_size_stride(primals_103, (1920, ), (1, ))
    assert_size_stride(primals_104, (1920, ), (1, ))
    assert_size_stride(primals_105, (640, ), (1, ))
    assert_size_stride(primals_106, (640, ), (1, ))
    assert_size_stride(primals_107, (1920, ), (1, ))
    assert_size_stride(primals_108, (1920, ), (1, ))
    assert_size_stride(primals_109, (1920, ), (1, ))
    assert_size_stride(primals_110, (1920, ), (1, ))
    assert_size_stride(primals_111, (640, ), (1, ))
    assert_size_stride(primals_112, (640, ), (1, ))
    assert_size_stride(primals_113, (2560, ), (1, ))
    assert_size_stride(primals_114, (2560, ), (1, ))
    assert_size_stride(primals_115, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_116, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_117, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_118, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_119, (192, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_120, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_121, (192, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_122, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_123, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_124, (160, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_125, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_126, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_127, (640, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_128, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_129, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_130, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_131, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_132, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_133, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_134, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_135, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_136, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_137, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_138, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_139, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_140, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_141, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_142, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_143, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_144, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_145, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_146, (640, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_147, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_148, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_149, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_150, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_151, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_152, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_153, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_154, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_155, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_156, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_157, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_158, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_159, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_160, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_161, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_162, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_163, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_164, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_165, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_166, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_167, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_168, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_169, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_170, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_171, (2560, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_172, (1000, 2560), (2560, 1))
    assert_size_stride(primals_173, (1000, ), (1, ))
    assert_size_stride(primals_174, (), ())
    assert_size_stride(primals_175, (32, ), (1, ))
    assert_size_stride(primals_176, (32, ), (1, ))
    assert_size_stride(primals_177, (), ())
    assert_size_stride(primals_178, (128, ), (1, ))
    assert_size_stride(primals_179, (128, ), (1, ))
    assert_size_stride(primals_180, (), ())
    assert_size_stride(primals_181, (128, ), (1, ))
    assert_size_stride(primals_182, (128, ), (1, ))
    assert_size_stride(primals_183, (), ())
    assert_size_stride(primals_184, (128, ), (1, ))
    assert_size_stride(primals_185, (128, ), (1, ))
    assert_size_stride(primals_186, (), ())
    assert_size_stride(primals_187, (192, ), (1, ))
    assert_size_stride(primals_188, (192, ), (1, ))
    assert_size_stride(primals_189, (), ())
    assert_size_stride(primals_190, (192, ), (1, ))
    assert_size_stride(primals_191, (192, ), (1, ))
    assert_size_stride(primals_192, (), ())
    assert_size_stride(primals_193, (192, ), (1, ))
    assert_size_stride(primals_194, (192, ), (1, ))
    assert_size_stride(primals_195, (), ())
    assert_size_stride(primals_196, (192, ), (1, ))
    assert_size_stride(primals_197, (192, ), (1, ))
    assert_size_stride(primals_198, (), ())
    assert_size_stride(primals_199, (192, ), (1, ))
    assert_size_stride(primals_200, (192, ), (1, ))
    assert_size_stride(primals_201, (), ())
    assert_size_stride(primals_202, (160, ), (1, ))
    assert_size_stride(primals_203, (160, ), (1, ))
    assert_size_stride(primals_204, (), ())
    assert_size_stride(primals_205, (160, ), (1, ))
    assert_size_stride(primals_206, (160, ), (1, ))
    assert_size_stride(primals_207, (), ())
    assert_size_stride(primals_208, (640, ), (1, ))
    assert_size_stride(primals_209, (640, ), (1, ))
    assert_size_stride(primals_210, (), ())
    assert_size_stride(primals_211, (640, ), (1, ))
    assert_size_stride(primals_212, (640, ), (1, ))
    assert_size_stride(primals_213, (), ())
    assert_size_stride(primals_214, (160, ), (1, ))
    assert_size_stride(primals_215, (160, ), (1, ))
    assert_size_stride(primals_216, (), ())
    assert_size_stride(primals_217, (160, ), (1, ))
    assert_size_stride(primals_218, (160, ), (1, ))
    assert_size_stride(primals_219, (), ())
    assert_size_stride(primals_220, (640, ), (1, ))
    assert_size_stride(primals_221, (640, ), (1, ))
    assert_size_stride(primals_222, (), ())
    assert_size_stride(primals_223, (160, ), (1, ))
    assert_size_stride(primals_224, (160, ), (1, ))
    assert_size_stride(primals_225, (), ())
    assert_size_stride(primals_226, (160, ), (1, ))
    assert_size_stride(primals_227, (160, ), (1, ))
    assert_size_stride(primals_228, (), ())
    assert_size_stride(primals_229, (640, ), (1, ))
    assert_size_stride(primals_230, (640, ), (1, ))
    assert_size_stride(primals_231, (), ())
    assert_size_stride(primals_232, (160, ), (1, ))
    assert_size_stride(primals_233, (160, ), (1, ))
    assert_size_stride(primals_234, (), ())
    assert_size_stride(primals_235, (160, ), (1, ))
    assert_size_stride(primals_236, (160, ), (1, ))
    assert_size_stride(primals_237, (), ())
    assert_size_stride(primals_238, (640, ), (1, ))
    assert_size_stride(primals_239, (640, ), (1, ))
    assert_size_stride(primals_240, (), ())
    assert_size_stride(primals_241, (160, ), (1, ))
    assert_size_stride(primals_242, (160, ), (1, ))
    assert_size_stride(primals_243, (), ())
    assert_size_stride(primals_244, (160, ), (1, ))
    assert_size_stride(primals_245, (160, ), (1, ))
    assert_size_stride(primals_246, (), ())
    assert_size_stride(primals_247, (640, ), (1, ))
    assert_size_stride(primals_248, (640, ), (1, ))
    assert_size_stride(primals_249, (), ())
    assert_size_stride(primals_250, (160, ), (1, ))
    assert_size_stride(primals_251, (160, ), (1, ))
    assert_size_stride(primals_252, (), ())
    assert_size_stride(primals_253, (160, ), (1, ))
    assert_size_stride(primals_254, (160, ), (1, ))
    assert_size_stride(primals_255, (), ())
    assert_size_stride(primals_256, (640, ), (1, ))
    assert_size_stride(primals_257, (640, ), (1, ))
    assert_size_stride(primals_258, (), ())
    assert_size_stride(primals_259, (1920, ), (1, ))
    assert_size_stride(primals_260, (1920, ), (1, ))
    assert_size_stride(primals_261, (), ())
    assert_size_stride(primals_262, (1920, ), (1, ))
    assert_size_stride(primals_263, (1920, ), (1, ))
    assert_size_stride(primals_264, (), ())
    assert_size_stride(primals_265, (640, ), (1, ))
    assert_size_stride(primals_266, (640, ), (1, ))
    assert_size_stride(primals_267, (), ())
    assert_size_stride(primals_268, (640, ), (1, ))
    assert_size_stride(primals_269, (640, ), (1, ))
    assert_size_stride(primals_270, (), ())
    assert_size_stride(primals_271, (1920, ), (1, ))
    assert_size_stride(primals_272, (1920, ), (1, ))
    assert_size_stride(primals_273, (), ())
    assert_size_stride(primals_274, (1920, ), (1, ))
    assert_size_stride(primals_275, (1920, ), (1, ))
    assert_size_stride(primals_276, (), ())
    assert_size_stride(primals_277, (640, ), (1, ))
    assert_size_stride(primals_278, (640, ), (1, ))
    assert_size_stride(primals_279, (), ())
    assert_size_stride(primals_280, (1920, ), (1, ))
    assert_size_stride(primals_281, (1920, ), (1, ))
    assert_size_stride(primals_282, (), ())
    assert_size_stride(primals_283, (1920, ), (1, ))
    assert_size_stride(primals_284, (1920, ), (1, ))
    assert_size_stride(primals_285, (), ())
    assert_size_stride(primals_286, (640, ), (1, ))
    assert_size_stride(primals_287, (640, ), (1, ))
    assert_size_stride(primals_288, (), ())
    assert_size_stride(primals_289, (1920, ), (1, ))
    assert_size_stride(primals_290, (1920, ), (1, ))
    assert_size_stride(primals_291, (), ())
    assert_size_stride(primals_292, (1920, ), (1, ))
    assert_size_stride(primals_293, (1920, ), (1, ))
    assert_size_stride(primals_294, (), ())
    assert_size_stride(primals_295, (640, ), (1, ))
    assert_size_stride(primals_296, (640, ), (1, ))
    assert_size_stride(primals_297, (), ())
    assert_size_stride(primals_298, (1920, ), (1, ))
    assert_size_stride(primals_299, (1920, ), (1, ))
    assert_size_stride(primals_300, (), ())
    assert_size_stride(primals_301, (1920, ), (1, ))
    assert_size_stride(primals_302, (1920, ), (1, ))
    assert_size_stride(primals_303, (), ())
    assert_size_stride(primals_304, (640, ), (1, ))
    assert_size_stride(primals_305, (640, ), (1, ))
    assert_size_stride(primals_306, (), ())
    assert_size_stride(primals_307, (1920, ), (1, ))
    assert_size_stride(primals_308, (1920, ), (1, ))
    assert_size_stride(primals_309, (), ())
    assert_size_stride(primals_310, (1920, ), (1, ))
    assert_size_stride(primals_311, (1920, ), (1, ))
    assert_size_stride(primals_312, (), ())
    assert_size_stride(primals_313, (640, ), (1, ))
    assert_size_stride(primals_314, (640, ), (1, ))
    assert_size_stride(primals_315, (), ())
    assert_size_stride(primals_316, (1920, ), (1, ))
    assert_size_stride(primals_317, (1920, ), (1, ))
    assert_size_stride(primals_318, (), ())
    assert_size_stride(primals_319, (1920, ), (1, ))
    assert_size_stride(primals_320, (1920, ), (1, ))
    assert_size_stride(primals_321, (), ())
    assert_size_stride(primals_322, (640, ), (1, ))
    assert_size_stride(primals_323, (640, ), (1, ))
    assert_size_stride(primals_324, (), ())
    assert_size_stride(primals_325, (1920, ), (1, ))
    assert_size_stride(primals_326, (1920, ), (1, ))
    assert_size_stride(primals_327, (), ())
    assert_size_stride(primals_328, (1920, ), (1, ))
    assert_size_stride(primals_329, (1920, ), (1, ))
    assert_size_stride(primals_330, (), ())
    assert_size_stride(primals_331, (640, ), (1, ))
    assert_size_stride(primals_332, (640, ), (1, ))
    assert_size_stride(primals_333, (), ())
    assert_size_stride(primals_334, (1920, ), (1, ))
    assert_size_stride(primals_335, (1920, ), (1, ))
    assert_size_stride(primals_336, (), ())
    assert_size_stride(primals_337, (1920, ), (1, ))
    assert_size_stride(primals_338, (1920, ), (1, ))
    assert_size_stride(primals_339, (), ())
    assert_size_stride(primals_340, (640, ), (1, ))
    assert_size_stride(primals_341, (640, ), (1, ))
    assert_size_stride(primals_342, (), ())
    assert_size_stride(primals_343, (2560, ), (1, ))
    assert_size_stride(primals_344, (2560, ), (1, ))
    assert_size_stride(primals_345, (8, 3, 256, 256), (196608, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_115, buf0, 96, 9, grid=grid(96, 9), stream=stream0)
        del primals_115
        buf1 = empty_strided((128, 32, 3, 3), (288, 1, 96, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_116, buf1, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_116
        buf2 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_117, buf2, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_117
        buf3 = empty_strided((192, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_119, buf3, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del primals_119
        buf4 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_120, buf4, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_120
        buf5 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_122, buf5, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_122
        buf6 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_123, buf6, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_123
        buf7 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_125, buf7, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_125
        buf8 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_129, buf8, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_129
        buf9 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_132, buf9, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_132
        buf10 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_135, buf10, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_135
        buf11 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_138, buf11, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_138
        buf12 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_141, buf12, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_141
        buf13 = empty_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_345, buf13, 24, 65536, grid=grid(24, 65536), stream=stream0)
        del primals_345
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 32, 128, 128), (524288, 16384, 128, 1))
        buf15 = empty_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(buf14, buf15, 256, 16384, grid=grid(256, 16384), stream=stream0)
        buf16 = empty_strided((1, 32, 1, 1, 1024), (32768, 1, 32768, 32768, 32), device='cuda', dtype=torch.float32)
        buf17 = empty_strided((1, 32, 1, 1, 1024), (32768, 1, 32768, 32768, 32), device='cuda', dtype=torch.float32)
        buf18 = empty_strided((1, 32, 1, 1, 1024), (32768, 1, 32768, 32768, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf15, buf16, buf17, buf18, 32768, 128, grid=grid(32768), stream=stream0)
        buf19 = empty_strided((1, 32, 1, 1, 8), (256, 1, 256, 256, 32), device='cuda', dtype=torch.float32)
        buf20 = empty_strided((1, 32, 1, 1, 8), (256, 1, 256, 256, 32), device='cuda', dtype=torch.float32)
        buf21 = empty_strided((1, 32, 1, 1, 8), (256, 1, 256, 256, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf16, buf17, buf18, buf19, buf20, buf21, 256, 128, grid=grid(256), stream=stream0)
        buf22 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf23 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf25 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_10.run(buf19, buf20, buf21, primals_175, primals_176, buf22, buf23, buf25, primals_175, primals_176, 32, 8, grid=grid(32), stream=stream0)
        del primals_175
        del primals_176
        buf26 = reinterpret_tensor(buf14, (8, 32, 128, 128), (524288, 1, 4096, 32), 0); del buf14  # reuse
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_11.run(buf15, buf22, buf23, primals_1, primals_2, buf26, 4194304, grid=grid(4194304), stream=stream0)
        del buf23
        del primals_2
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf28 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf27, buf28, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        buf29 = reinterpret_tensor(buf18, (1, 128, 1, 1, 256), (32768, 1, 32768, 32768, 128), 0); del buf18  # reuse
        buf30 = reinterpret_tensor(buf17, (1, 128, 1, 1, 256), (32768, 1, 32768, 32768, 128), 0); del buf17  # reuse
        buf31 = reinterpret_tensor(buf16, (1, 128, 1, 1, 256), (32768, 1, 32768, 32768, 128), 0); del buf16  # reuse
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf28, buf29, buf30, buf31, 32768, 128, grid=grid(32768), stream=stream0)
        buf32 = reinterpret_tensor(buf21, (1, 128, 1, 1, 2), (256, 1, 256, 256, 128), 0); del buf21  # reuse
        buf33 = reinterpret_tensor(buf20, (1, 128, 1, 1, 2), (256, 1, 256, 256, 128), 0); del buf20  # reuse
        buf34 = reinterpret_tensor(buf19, (1, 128, 1, 1, 2), (256, 1, 256, 256, 128), 0); del buf19  # reuse
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf29, buf30, buf31, buf32, buf33, buf34, 256, 128, grid=grid(256), stream=stream0)
        buf35 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf36 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf38 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf32, buf33, buf34, primals_178, primals_179, buf35, buf36, buf38, primals_178, primals_179, 128, 2, grid=grid(128), stream=stream0)
        del primals_178
        del primals_179
        buf39 = reinterpret_tensor(buf27, (8, 128, 64, 64), (524288, 1, 8192, 128), 0); del buf27  # reuse
        # Source Nodes: [x_11, x_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_16.run(buf28, buf35, buf36, primals_3, primals_4, buf39, 4194304, grid=grid(4194304), stream=stream0)
        del primals_4
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf41 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf40, buf41, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        buf42 = buf31; del buf31  # reuse
        buf43 = buf30; del buf30  # reuse
        buf44 = buf29; del buf29  # reuse
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf41, buf42, buf43, buf44, 32768, 128, grid=grid(32768), stream=stream0)
        buf45 = buf34; del buf34  # reuse
        buf46 = buf33; del buf33  # reuse
        buf47 = buf32; del buf32  # reuse
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf42, buf43, buf44, buf45, buf46, buf47, 256, 128, grid=grid(256), stream=stream0)
        buf48 = buf36; del buf36  # reuse
        buf49 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf51 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf45, buf46, buf47, primals_181, primals_182, buf48, buf49, buf51, primals_181, primals_182, 128, 2, grid=grid(128), stream=stream0)
        del primals_181
        del primals_182
        # Source Nodes: [x_20], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf26, primals_118, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf53 = reinterpret_tensor(buf40, (8, 128, 64, 64), (524288, 1, 8192, 128), 0); del buf40  # reuse
        # Source Nodes: [x_20], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf52, buf53, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        buf54 = buf44; del buf44  # reuse
        buf55 = buf43; del buf43  # reuse
        buf56 = buf42; del buf42  # reuse
        # Source Nodes: [x_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf53, buf54, buf55, buf56, 32768, 128, grid=grid(32768), stream=stream0)
        buf57 = buf47; del buf47  # reuse
        buf58 = buf46; del buf46  # reuse
        buf59 = buf45; del buf45  # reuse
        # Source Nodes: [x_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf54, buf55, buf56, buf57, buf58, buf59, 256, 128, grid=grid(256), stream=stream0)
        del buf54
        del buf55
        del buf56
        buf60 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf61 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf63 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf57, buf58, buf59, primals_184, primals_185, buf60, buf61, buf63, primals_184, primals_185, 128, 2, grid=grid(128), stream=stream0)
        del buf57
        del buf58
        del buf59
        del primals_184
        del primals_185
        buf64 = reinterpret_tensor(buf52, (8, 128, 64, 64), (524288, 1, 8192, 128), 0); del buf52  # reuse
        buf65 = buf64; del buf64  # reuse
        # Source Nodes: [shortcut_1, x_13, x_21, x_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_17.run(buf65, buf41, buf48, buf49, primals_5, primals_6, buf53, buf60, buf61, primals_7, primals_8, 4194304, grid=grid(4194304), stream=stream0)
        del buf49
        del buf61
        del primals_6
        del primals_8
        # Source Nodes: [x_26], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, buf3, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 192, 32, 32), (196608, 1024, 32, 1))
        buf67 = empty_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_26], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf66, buf67, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        buf68 = empty_strided((1, 192, 1, 1, 64), (12288, 1, 12288, 12288, 192), device='cuda', dtype=torch.float32)
        buf69 = empty_strided((1, 192, 1, 1, 64), (12288, 1, 12288, 12288, 192), device='cuda', dtype=torch.float32)
        buf70 = empty_strided((1, 192, 1, 1, 64), (12288, 1, 12288, 12288, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_27], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf67, buf68, buf69, buf70, 12288, 128, grid=grid(12288), stream=stream0)
        buf71 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf72 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf74 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_27], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf68, buf69, buf70, primals_187, primals_188, buf71, buf72, buf74, primals_187, primals_188, 192, 64, grid=grid(192), stream=stream0)
        del primals_187
        del primals_188
        buf75 = reinterpret_tensor(buf66, (8, 192, 32, 32), (196608, 1, 6144, 192), 0); del buf66  # reuse
        # Source Nodes: [x_27, x_31], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_21.run(buf67, buf71, buf72, primals_9, primals_10, buf75, 1572864, grid=grid(1572864), stream=stream0)
        del primals_10
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 192, 32, 32), (196608, 1024, 32, 1))
        buf77 = empty_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf76, buf77, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        buf78 = buf70; del buf70  # reuse
        buf79 = buf69; del buf69  # reuse
        buf80 = buf68; del buf68  # reuse
        # Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf77, buf78, buf79, buf80, 12288, 128, grid=grid(12288), stream=stream0)
        buf81 = buf72; del buf72  # reuse
        buf82 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf84 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf78, buf79, buf80, primals_190, primals_191, buf81, buf82, buf84, primals_190, primals_191, 192, 64, grid=grid(192), stream=stream0)
        del primals_190
        del primals_191
        # Source Nodes: [x_40], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf65, primals_121, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (8, 192, 32, 32), (196608, 1024, 32, 1))
        buf86 = reinterpret_tensor(buf76, (8, 192, 32, 32), (196608, 1, 6144, 192), 0); del buf76  # reuse
        # Source Nodes: [x_40], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf85, buf86, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        buf87 = buf80; del buf80  # reuse
        buf88 = buf79; del buf79  # reuse
        buf89 = buf78; del buf78  # reuse
        # Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf86, buf87, buf88, buf89, 12288, 128, grid=grid(12288), stream=stream0)
        buf90 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf91 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf93 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf87, buf88, buf89, primals_193, primals_194, buf90, buf91, buf93, primals_193, primals_194, 192, 64, grid=grid(192), stream=stream0)
        del primals_193
        del primals_194
        buf94 = reinterpret_tensor(buf85, (8, 192, 32, 32), (196608, 1, 6144, 192), 0); del buf85  # reuse
        buf95 = buf94; del buf94  # reuse
        # Source Nodes: [shortcut_2, x_33, x_41, x_45], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf95, buf77, buf81, buf82, primals_11, primals_12, buf86, buf90, buf91, primals_13, primals_14, 1572864, grid=grid(1572864), stream=stream0)
        del primals_12
        del primals_14
        # Source Nodes: [x_46], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 192, 32, 32), (196608, 1024, 32, 1))
        buf97 = empty_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_46], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf96, buf97, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        buf98 = buf89; del buf89  # reuse
        buf99 = buf88; del buf88  # reuse
        buf100 = buf87; del buf87  # reuse
        # Source Nodes: [x_47], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf97, buf98, buf99, buf100, 12288, 128, grid=grid(12288), stream=stream0)
        buf101 = buf91; del buf91  # reuse
        buf102 = buf82; del buf82  # reuse
        buf104 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_47], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf98, buf99, buf100, primals_196, primals_197, buf101, buf102, buf104, primals_196, primals_197, 192, 64, grid=grid(192), stream=stream0)
        del primals_196
        del primals_197
        buf105 = reinterpret_tensor(buf96, (8, 192, 32, 32), (196608, 1, 6144, 192), 0); del buf96  # reuse
        # Source Nodes: [x_47, x_51], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_21.run(buf97, buf101, buf102, primals_15, primals_16, buf105, 1572864, grid=grid(1572864), stream=stream0)
        del primals_16
        # Source Nodes: [x_52], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 192, 32, 32), (196608, 1024, 32, 1))
        buf107 = empty_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_52], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf106, buf107, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        buf108 = buf99; del buf99  # reuse
        buf109 = buf98; del buf98  # reuse
        buf110 = buf100; del buf100  # reuse
        # Source Nodes: [x_53], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf107, buf108, buf109, buf110, 12288, 128, grid=grid(12288), stream=stream0)
        buf111 = buf102; del buf102  # reuse
        buf112 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf114 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_53], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf108, buf109, buf110, primals_199, primals_200, buf111, buf112, buf114, primals_199, primals_200, 192, 64, grid=grid(192), stream=stream0)
        del buf108
        del buf109
        del buf110
        del primals_199
        del primals_200
        buf115 = reinterpret_tensor(buf106, (8, 192, 32, 32), (196608, 1, 6144, 192), 0); del buf106  # reuse
        # Source Nodes: [shortcut_3, x_53, x_60], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_23.run(buf107, buf111, buf112, primals_17, primals_18, buf95, buf115, 1572864, grid=grid(1572864), stream=stream0)
        del buf112
        del primals_18
        # Source Nodes: [x_61], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 160, 32, 32), (163840, 1024, 32, 1))
        buf117 = empty_strided((8, 160, 32, 32), (163840, 1, 5120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf116, buf117, 1280, 1024, grid=grid(1280, 1024), stream=stream0)
        buf118 = empty_strided((1, 160, 1, 1, 64), (10240, 1, 10240, 10240, 160), device='cuda', dtype=torch.float32)
        buf119 = empty_strided((1, 160, 1, 1, 64), (10240, 1, 10240, 10240, 160), device='cuda', dtype=torch.float32)
        buf120 = empty_strided((1, 160, 1, 1, 64), (10240, 1, 10240, 10240, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf117, buf118, buf119, buf120, 10240, 128, grid=grid(10240), stream=stream0)
        buf121 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf122 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf124 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf118, buf119, buf120, primals_202, primals_203, buf121, buf122, buf124, primals_202, primals_203, 160, 64, grid=grid(160), stream=stream0)
        del primals_202
        del primals_203
        buf125 = reinterpret_tensor(buf116, (8, 160, 32, 32), (163840, 1, 5120, 160), 0); del buf116  # reuse
        # Source Nodes: [x_62, x_66], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_27.run(buf117, buf121, buf122, primals_19, primals_20, buf125, 1310720, grid=grid(1310720), stream=stream0)
        del primals_20
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, buf7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf127 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf126, buf127, 1280, 256, grid=grid(1280, 256), stream=stream0)
        buf128 = empty_strided((1, 160, 1, 1, 16), (2560, 1, 2560, 2560, 160), device='cuda', dtype=torch.float32)
        buf129 = empty_strided((1, 160, 1, 1, 16), (2560, 1, 2560, 2560, 160), device='cuda', dtype=torch.float32)
        buf130 = empty_strided((1, 160, 1, 1, 16), (2560, 1, 2560, 2560, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf127, buf128, buf129, buf130, 2560, 128, grid=grid(2560), stream=stream0)
        buf131 = buf122; del buf122  # reuse
        buf132 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf134 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf128, buf129, buf130, primals_205, primals_206, buf131, buf132, buf134, primals_205, primals_206, 160, 16, grid=grid(160), stream=stream0)
        del primals_205
        del primals_206
        buf135 = reinterpret_tensor(buf126, (8, 160, 16, 16), (40960, 1, 2560, 160), 0); del buf126  # reuse
        # Source Nodes: [x_68, x_72], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_31.run(buf127, buf131, buf132, primals_21, primals_22, buf135, 327680, grid=grid(327680), stream=stream0)
        del primals_22
        # Source Nodes: [x_75], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf137 = empty_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_75], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_32.run(buf136, buf137, 5120, 256, grid=grid(5120, 256), stream=stream0)
        buf138 = reinterpret_tensor(buf120, (1, 640, 1, 1, 16), (10240, 1, 10240, 10240, 640), 0); del buf120  # reuse
        buf139 = reinterpret_tensor(buf119, (1, 640, 1, 1, 16), (10240, 1, 10240, 10240, 640), 0); del buf119  # reuse
        buf140 = reinterpret_tensor(buf118, (1, 640, 1, 1, 16), (10240, 1, 10240, 10240, 640), 0); del buf118  # reuse
        # Source Nodes: [x_76], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf137, buf138, buf139, buf140, 10240, 128, grid=grid(10240), stream=stream0)
        buf141 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf142 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf144 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_76], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf138, buf139, buf140, primals_208, primals_209, buf141, buf142, buf144, primals_208, primals_209, 640, 16, grid=grid(640), stream=stream0)
        del primals_208
        del primals_209
        # Source Nodes: [x_83], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf115, primals_127, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf146 = reinterpret_tensor(buf136, (8, 640, 16, 16), (163840, 1, 10240, 640), 0); del buf136  # reuse
        # Source Nodes: [x_83], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_32.run(buf145, buf146, 5120, 256, grid=grid(5120, 256), stream=stream0)
        buf147 = buf140; del buf140  # reuse
        buf148 = buf139; del buf139  # reuse
        buf149 = buf138; del buf138  # reuse
        # Source Nodes: [x_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf146, buf147, buf148, buf149, 10240, 128, grid=grid(10240), stream=stream0)
        buf150 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf151 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf153 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf147, buf148, buf149, primals_211, primals_212, buf150, buf151, buf153, primals_211, primals_212, 640, 16, grid=grid(640), stream=stream0)
        del primals_211
        del primals_212
        buf154 = reinterpret_tensor(buf145, (8, 640, 16, 16), (163840, 1, 10240, 640), 0); del buf145  # reuse
        buf155 = buf154; del buf154  # reuse
        # Source Nodes: [shortcut_4, x_76, x_84, x_88], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_35.run(buf155, buf137, buf141, buf142, primals_23, primals_24, buf146, buf150, buf151, primals_25, primals_26, 1310720, grid=grid(1310720), stream=stream0)
        del primals_24
        del primals_26
        # Source Nodes: [x_89], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf157 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf156, buf157, 1280, 256, grid=grid(1280, 256), stream=stream0)
        buf158 = buf130; del buf130  # reuse
        buf159 = buf129; del buf129  # reuse
        buf160 = buf128; del buf128  # reuse
        # Source Nodes: [x_90], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf157, buf158, buf159, buf160, 2560, 128, grid=grid(2560), stream=stream0)
        buf161 = buf132; del buf132  # reuse
        buf162 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf164 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_90], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf158, buf159, buf160, primals_214, primals_215, buf161, buf162, buf164, primals_214, primals_215, 160, 16, grid=grid(160), stream=stream0)
        del primals_214
        del primals_215
        buf165 = reinterpret_tensor(buf156, (8, 160, 16, 16), (40960, 1, 2560, 160), 0); del buf156  # reuse
        # Source Nodes: [x_90, x_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_31.run(buf157, buf161, buf162, primals_27, primals_28, buf165, 327680, grid=grid(327680), stream=stream0)
        del primals_28
        # Source Nodes: [x_95], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf167 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_95], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf166, buf167, 1280, 256, grid=grid(1280, 256), stream=stream0)
        buf168 = buf160; del buf160  # reuse
        buf169 = buf159; del buf159  # reuse
        buf170 = buf158; del buf158  # reuse
        # Source Nodes: [x_96], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf167, buf168, buf169, buf170, 2560, 128, grid=grid(2560), stream=stream0)
        buf171 = buf162; del buf162  # reuse
        buf172 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf174 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_96], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf168, buf169, buf170, primals_217, primals_218, buf171, buf172, buf174, primals_217, primals_218, 160, 16, grid=grid(160), stream=stream0)
        del primals_217
        del primals_218
        buf175 = reinterpret_tensor(buf166, (8, 160, 16, 16), (40960, 1, 2560, 160), 0); del buf166  # reuse
        # Source Nodes: [x_100, x_96], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_31.run(buf167, buf171, buf172, primals_29, primals_30, buf175, 327680, grid=grid(327680), stream=stream0)
        del primals_30
        # Source Nodes: [x_103], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf177 = empty_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_103], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_32.run(buf176, buf177, 5120, 256, grid=grid(5120, 256), stream=stream0)
        buf178 = buf149; del buf149  # reuse
        buf179 = buf148; del buf148  # reuse
        buf180 = buf147; del buf147  # reuse
        # Source Nodes: [x_104], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf177, buf178, buf179, buf180, 10240, 128, grid=grid(10240), stream=stream0)
        buf181 = buf151; del buf151  # reuse
        buf182 = buf142; del buf142  # reuse
        buf184 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_104], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf178, buf179, buf180, primals_220, primals_221, buf181, buf182, buf184, primals_220, primals_221, 640, 16, grid=grid(640), stream=stream0)
        del primals_220
        del primals_221
        buf185 = reinterpret_tensor(buf176, (8, 640, 16, 16), (163840, 1, 10240, 640), 0); del buf176  # reuse
        # Source Nodes: [shortcut_5, x_104, x_111], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_36.run(buf177, buf181, buf182, primals_31, primals_32, buf155, buf185, 1310720, grid=grid(1310720), stream=stream0)
        del primals_32
        # Source Nodes: [x_112], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, primals_131, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf187 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_112], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf186, buf187, 1280, 256, grid=grid(1280, 256), stream=stream0)
        buf188 = buf170; del buf170  # reuse
        buf189 = buf169; del buf169  # reuse
        buf190 = buf168; del buf168  # reuse
        # Source Nodes: [x_113], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf187, buf188, buf189, buf190, 2560, 128, grid=grid(2560), stream=stream0)
        buf191 = buf172; del buf172  # reuse
        buf192 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf194 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_113], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf188, buf189, buf190, primals_223, primals_224, buf191, buf192, buf194, primals_223, primals_224, 160, 16, grid=grid(160), stream=stream0)
        del primals_223
        del primals_224
        buf195 = reinterpret_tensor(buf186, (8, 160, 16, 16), (40960, 1, 2560, 160), 0); del buf186  # reuse
        # Source Nodes: [x_113, x_117], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_31.run(buf187, buf191, buf192, primals_33, primals_34, buf195, 327680, grid=grid(327680), stream=stream0)
        del primals_34
        # Source Nodes: [x_118], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf197 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_118], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf196, buf197, 1280, 256, grid=grid(1280, 256), stream=stream0)
        buf198 = buf190; del buf190  # reuse
        buf199 = buf189; del buf189  # reuse
        buf200 = buf188; del buf188  # reuse
        # Source Nodes: [x_119], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf197, buf198, buf199, buf200, 2560, 128, grid=grid(2560), stream=stream0)
        buf201 = buf192; del buf192  # reuse
        buf202 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf204 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_119], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf198, buf199, buf200, primals_226, primals_227, buf201, buf202, buf204, primals_226, primals_227, 160, 16, grid=grid(160), stream=stream0)
        del primals_226
        del primals_227
        buf205 = reinterpret_tensor(buf196, (8, 160, 16, 16), (40960, 1, 2560, 160), 0); del buf196  # reuse
        # Source Nodes: [x_119, x_123], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_31.run(buf197, buf201, buf202, primals_35, primals_36, buf205, 327680, grid=grid(327680), stream=stream0)
        del primals_36
        # Source Nodes: [x_126], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf207 = empty_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_126], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_32.run(buf206, buf207, 5120, 256, grid=grid(5120, 256), stream=stream0)
        buf208 = buf180; del buf180  # reuse
        buf209 = buf179; del buf179  # reuse
        buf210 = buf178; del buf178  # reuse
        # Source Nodes: [x_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf207, buf208, buf209, buf210, 10240, 128, grid=grid(10240), stream=stream0)
        buf211 = buf182; del buf182  # reuse
        buf212 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf214 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf208, buf209, buf210, primals_229, primals_230, buf211, buf212, buf214, primals_229, primals_230, 640, 16, grid=grid(640), stream=stream0)
        del primals_229
        del primals_230
        buf215 = reinterpret_tensor(buf206, (8, 640, 16, 16), (163840, 1, 10240, 640), 0); del buf206  # reuse
        # Source Nodes: [shortcut_6, x_127, x_134], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_36.run(buf207, buf211, buf212, primals_37, primals_38, buf185, buf215, 1310720, grid=grid(1310720), stream=stream0)
        del primals_38
        # Source Nodes: [x_135], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_134, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf217 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf216, buf217, 1280, 256, grid=grid(1280, 256), stream=stream0)
        buf218 = buf200; del buf200  # reuse
        buf219 = buf199; del buf199  # reuse
        buf220 = buf198; del buf198  # reuse
        # Source Nodes: [x_136], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf217, buf218, buf219, buf220, 2560, 128, grid=grid(2560), stream=stream0)
        buf221 = buf202; del buf202  # reuse
        buf222 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf224 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_136], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf218, buf219, buf220, primals_232, primals_233, buf221, buf222, buf224, primals_232, primals_233, 160, 16, grid=grid(160), stream=stream0)
        del primals_232
        del primals_233
        buf225 = reinterpret_tensor(buf216, (8, 160, 16, 16), (40960, 1, 2560, 160), 0); del buf216  # reuse
        # Source Nodes: [x_136, x_140], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_31.run(buf217, buf221, buf222, primals_39, primals_40, buf225, 327680, grid=grid(327680), stream=stream0)
        del primals_40
        # Source Nodes: [x_141], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf225, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf227 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_141], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf226, buf227, 1280, 256, grid=grid(1280, 256), stream=stream0)
        buf228 = buf220; del buf220  # reuse
        buf229 = buf219; del buf219  # reuse
        buf230 = buf218; del buf218  # reuse
        # Source Nodes: [x_142], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf227, buf228, buf229, buf230, 2560, 128, grid=grid(2560), stream=stream0)
        buf231 = buf222; del buf222  # reuse
        buf232 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf234 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_142], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf228, buf229, buf230, primals_235, primals_236, buf231, buf232, buf234, primals_235, primals_236, 160, 16, grid=grid(160), stream=stream0)
        del primals_235
        del primals_236
        buf235 = reinterpret_tensor(buf226, (8, 160, 16, 16), (40960, 1, 2560, 160), 0); del buf226  # reuse
        # Source Nodes: [x_142, x_146], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_31.run(buf227, buf231, buf232, primals_41, primals_42, buf235, 327680, grid=grid(327680), stream=stream0)
        del primals_42
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        buf236 = extern_kernels.convolution(buf235, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf237 = empty_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_32.run(buf236, buf237, 5120, 256, grid=grid(5120, 256), stream=stream0)
        buf238 = buf210; del buf210  # reuse
        buf239 = buf209; del buf209  # reuse
        buf240 = buf208; del buf208  # reuse
        # Source Nodes: [x_150], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf237, buf238, buf239, buf240, 10240, 128, grid=grid(10240), stream=stream0)
        buf241 = buf212; del buf212  # reuse
        buf242 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf244 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_150], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf238, buf239, buf240, primals_238, primals_239, buf241, buf242, buf244, primals_238, primals_239, 640, 16, grid=grid(640), stream=stream0)
        del primals_238
        del primals_239
        buf245 = reinterpret_tensor(buf236, (8, 640, 16, 16), (163840, 1, 10240, 640), 0); del buf236  # reuse
        # Source Nodes: [shortcut_7, x_150, x_157], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_36.run(buf237, buf241, buf242, primals_43, primals_44, buf215, buf245, 1310720, grid=grid(1310720), stream=stream0)
        del primals_44
        # Source Nodes: [x_158], Original ATen: [aten.convolution]
        buf246 = extern_kernels.convolution(buf245, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf246, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf247 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf246, buf247, 1280, 256, grid=grid(1280, 256), stream=stream0)
        buf248 = buf230; del buf230  # reuse
        buf249 = buf229; del buf229  # reuse
        buf250 = buf228; del buf228  # reuse
        # Source Nodes: [x_159], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf247, buf248, buf249, buf250, 2560, 128, grid=grid(2560), stream=stream0)
        buf251 = buf232; del buf232  # reuse
        buf252 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf254 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_159], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf248, buf249, buf250, primals_241, primals_242, buf251, buf252, buf254, primals_241, primals_242, 160, 16, grid=grid(160), stream=stream0)
        del primals_241
        del primals_242
        buf255 = reinterpret_tensor(buf246, (8, 160, 16, 16), (40960, 1, 2560, 160), 0); del buf246  # reuse
        # Source Nodes: [x_159, x_163], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_31.run(buf247, buf251, buf252, primals_45, primals_46, buf255, 327680, grid=grid(327680), stream=stream0)
        del primals_46
        # Source Nodes: [x_164], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf257 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_164], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf256, buf257, 1280, 256, grid=grid(1280, 256), stream=stream0)
        buf258 = buf250; del buf250  # reuse
        buf259 = buf249; del buf249  # reuse
        buf260 = buf248; del buf248  # reuse
        # Source Nodes: [x_165], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf257, buf258, buf259, buf260, 2560, 128, grid=grid(2560), stream=stream0)
        buf261 = buf252; del buf252  # reuse
        buf262 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf264 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_165], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf258, buf259, buf260, primals_244, primals_245, buf261, buf262, buf264, primals_244, primals_245, 160, 16, grid=grid(160), stream=stream0)
        del primals_244
        del primals_245
        buf265 = reinterpret_tensor(buf256, (8, 160, 16, 16), (40960, 1, 2560, 160), 0); del buf256  # reuse
        # Source Nodes: [x_165, x_169], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_31.run(buf257, buf261, buf262, primals_47, primals_48, buf265, 327680, grid=grid(327680), stream=stream0)
        del primals_48
        # Source Nodes: [x_172], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf265, primals_139, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf267 = empty_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_32.run(buf266, buf267, 5120, 256, grid=grid(5120, 256), stream=stream0)
        buf268 = buf240; del buf240  # reuse
        buf269 = buf239; del buf239  # reuse
        buf270 = buf238; del buf238  # reuse
        # Source Nodes: [x_173], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf267, buf268, buf269, buf270, 10240, 128, grid=grid(10240), stream=stream0)
        buf271 = buf242; del buf242  # reuse
        buf272 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf274 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_173], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf268, buf269, buf270, primals_247, primals_248, buf271, buf272, buf274, primals_247, primals_248, 640, 16, grid=grid(640), stream=stream0)
        del primals_247
        del primals_248
        buf275 = reinterpret_tensor(buf266, (8, 640, 16, 16), (163840, 1, 10240, 640), 0); del buf266  # reuse
        # Source Nodes: [shortcut_8, x_173, x_180], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_36.run(buf267, buf271, buf272, primals_49, primals_50, buf245, buf275, 1310720, grid=grid(1310720), stream=stream0)
        del primals_50
        # Source Nodes: [x_181], Original ATen: [aten.convolution]
        buf276 = extern_kernels.convolution(buf275, primals_140, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf277 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_181], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf276, buf277, 1280, 256, grid=grid(1280, 256), stream=stream0)
        buf278 = buf260; del buf260  # reuse
        buf279 = buf259; del buf259  # reuse
        buf280 = buf258; del buf258  # reuse
        # Source Nodes: [x_182], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf277, buf278, buf279, buf280, 2560, 128, grid=grid(2560), stream=stream0)
        buf281 = buf262; del buf262  # reuse
        buf282 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf284 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_182], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf278, buf279, buf280, primals_250, primals_251, buf281, buf282, buf284, primals_250, primals_251, 160, 16, grid=grid(160), stream=stream0)
        del primals_250
        del primals_251
        buf285 = reinterpret_tensor(buf276, (8, 160, 16, 16), (40960, 1, 2560, 160), 0); del buf276  # reuse
        # Source Nodes: [x_182, x_186], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_31.run(buf277, buf281, buf282, primals_51, primals_52, buf285, 327680, grid=grid(327680), stream=stream0)
        del primals_52
        # Source Nodes: [x_187], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf285, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf286, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf287 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_187], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf286, buf287, 1280, 256, grid=grid(1280, 256), stream=stream0)
        buf288 = buf280; del buf280  # reuse
        buf289 = buf279; del buf279  # reuse
        buf290 = buf278; del buf278  # reuse
        # Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf287, buf288, buf289, buf290, 2560, 128, grid=grid(2560), stream=stream0)
        buf291 = buf282; del buf282  # reuse
        buf292 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf294 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf288, buf289, buf290, primals_253, primals_254, buf291, buf292, buf294, primals_253, primals_254, 160, 16, grid=grid(160), stream=stream0)
        del primals_253
        del primals_254
        buf295 = reinterpret_tensor(buf286, (8, 160, 16, 16), (40960, 1, 2560, 160), 0); del buf286  # reuse
        # Source Nodes: [x_188, x_192], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_31.run(buf287, buf291, buf292, primals_53, primals_54, buf295, 327680, grid=grid(327680), stream=stream0)
        del buf292
        del primals_54
        # Source Nodes: [x_195], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf295, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf297 = empty_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_195], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_32.run(buf296, buf297, 5120, 256, grid=grid(5120, 256), stream=stream0)
        buf298 = buf270; del buf270  # reuse
        buf299 = buf269; del buf269  # reuse
        buf300 = buf268; del buf268  # reuse
        # Source Nodes: [x_196], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf297, buf298, buf299, buf300, 10240, 128, grid=grid(10240), stream=stream0)
        buf301 = buf272; del buf272  # reuse
        buf302 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf304 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_196], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf298, buf299, buf300, primals_256, primals_257, buf301, buf302, buf304, primals_256, primals_257, 640, 16, grid=grid(640), stream=stream0)
        del primals_256
        del primals_257
        buf305 = reinterpret_tensor(buf296, (8, 640, 16, 16), (163840, 1, 10240, 640), 0); del buf296  # reuse
        # Source Nodes: [shortcut_9, x_196, x_203], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_36.run(buf297, buf301, buf302, primals_55, primals_56, buf275, buf305, 1310720, grid=grid(1310720), stream=stream0)
        del primals_56
        # Source Nodes: [x_204], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, primals_143, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (8, 1920, 16, 16), (491520, 256, 16, 1))
        buf307 = empty_strided((8, 1920, 16, 16), (491520, 1, 30720, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_204], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf306, buf307, 15360, 256, grid=grid(15360, 256), stream=stream0)
        buf308 = empty_strided((1, 1920, 1, 1, 16), (30720, 1, 30720, 30720, 1920), device='cuda', dtype=torch.float32)
        buf309 = empty_strided((1, 1920, 1, 1, 16), (30720, 1, 30720, 30720, 1920), device='cuda', dtype=torch.float32)
        buf310 = empty_strided((1, 1920, 1, 1, 16), (30720, 1, 30720, 30720, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf307, buf308, buf309, buf310, 30720, 128, grid=grid(30720), stream=stream0)
        buf311 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf312 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf314 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf308, buf309, buf310, primals_259, primals_260, buf311, buf312, buf314, primals_259, primals_260, 1920, 16, grid=grid(1920), stream=stream0)
        del buf308
        del buf309
        del buf310
        del primals_259
        del primals_260
        buf315 = reinterpret_tensor(buf306, (8, 1920, 16, 16), (491520, 1, 30720, 1920), 0); del buf306  # reuse
        # Source Nodes: [x_205, x_209], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_40.run(buf307, buf311, buf312, primals_57, primals_58, buf315, 3932160, grid=grid(3932160), stream=stream0)
        del primals_58
        # Source Nodes: [x_210], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf315, primals_144, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf316, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf317 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_210], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf316, buf317, 15360, 64, grid=grid(15360, 64), stream=stream0)
        buf318 = empty_strided((1, 1920, 1, 1, 4), (7680, 1, 7680, 7680, 1920), device='cuda', dtype=torch.float32)
        buf319 = empty_strided((1, 1920, 1, 1, 4), (7680, 1, 7680, 7680, 1920), device='cuda', dtype=torch.float32)
        buf320 = empty_strided((1, 1920, 1, 1, 4), (7680, 1, 7680, 7680, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf317, buf318, buf319, buf320, 7680, 128, grid=grid(7680), stream=stream0)
        buf321 = buf312; del buf312  # reuse
        buf322 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf324 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf318, buf319, buf320, primals_262, primals_263, buf321, buf322, buf324, primals_262, primals_263, 1920, 4, grid=grid(1920), stream=stream0)
        del primals_262
        del primals_263
        buf325 = reinterpret_tensor(buf316, (8, 1920, 8, 8), (122880, 1, 15360, 1920), 0); del buf316  # reuse
        # Source Nodes: [x_211, x_215], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf317, buf321, buf322, primals_59, primals_60, buf325, 983040, grid=grid(983040), stream=stream0)
        del primals_60
        # Source Nodes: [x_218], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf325, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf327 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_218], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf326, buf327, 5120, 64, grid=grid(5120, 64), stream=stream0)
        buf328 = reinterpret_tensor(buf290, (1, 640, 1, 1, 4), (2560, 1, 2560, 2560, 640), 0); del buf290  # reuse
        buf329 = reinterpret_tensor(buf289, (1, 640, 1, 1, 4), (2560, 1, 2560, 2560, 640), 0); del buf289  # reuse
        buf330 = reinterpret_tensor(buf288, (1, 640, 1, 1, 4), (2560, 1, 2560, 2560, 640), 0); del buf288  # reuse
        # Source Nodes: [x_219], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf327, buf328, buf329, buf330, 2560, 128, grid=grid(2560), stream=stream0)
        buf331 = buf302; del buf302  # reuse
        buf332 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf334 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_219], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf328, buf329, buf330, primals_265, primals_266, buf331, buf332, buf334, primals_265, primals_266, 640, 4, grid=grid(640), stream=stream0)
        del primals_265
        del primals_266
        # Source Nodes: [x_226], Original ATen: [aten.convolution]
        buf335 = extern_kernels.convolution(buf305, primals_146, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf335, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf336 = reinterpret_tensor(buf326, (8, 640, 8, 8), (40960, 1, 5120, 640), 0); del buf326  # reuse
        # Source Nodes: [x_226], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf335, buf336, 5120, 64, grid=grid(5120, 64), stream=stream0)
        buf337 = buf330; del buf330  # reuse
        buf338 = buf329; del buf329  # reuse
        buf339 = buf328; del buf328  # reuse
        # Source Nodes: [x_227], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf336, buf337, buf338, buf339, 2560, 128, grid=grid(2560), stream=stream0)
        buf340 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf341 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf343 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_227], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf337, buf338, buf339, primals_268, primals_269, buf340, buf341, buf343, primals_268, primals_269, 640, 4, grid=grid(640), stream=stream0)
        del primals_268
        del primals_269
        buf344 = reinterpret_tensor(buf335, (8, 640, 8, 8), (40960, 1, 5120, 640), 0); del buf335  # reuse
        buf345 = buf344; del buf344  # reuse
        # Source Nodes: [shortcut_10, x_219, x_227, x_231], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_48.run(buf345, buf327, buf331, buf332, primals_61, primals_62, buf336, buf340, buf341, primals_63, primals_64, 327680, grid=grid(327680), stream=stream0)
        del primals_62
        del primals_64
        # Source Nodes: [x_232], Original ATen: [aten.convolution]
        buf346 = extern_kernels.convolution(buf345, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf347 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_232], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf346, buf347, 15360, 64, grid=grid(15360, 64), stream=stream0)
        buf348 = buf320; del buf320  # reuse
        buf349 = buf319; del buf319  # reuse
        buf350 = buf318; del buf318  # reuse
        # Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf347, buf348, buf349, buf350, 7680, 128, grid=grid(7680), stream=stream0)
        buf351 = buf322; del buf322  # reuse
        buf352 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf354 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf348, buf349, buf350, primals_271, primals_272, buf351, buf352, buf354, primals_271, primals_272, 1920, 4, grid=grid(1920), stream=stream0)
        del primals_271
        del primals_272
        buf355 = reinterpret_tensor(buf346, (8, 1920, 8, 8), (122880, 1, 15360, 1920), 0); del buf346  # reuse
        # Source Nodes: [x_233, x_237], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf347, buf351, buf352, primals_65, primals_66, buf355, 983040, grid=grid(983040), stream=stream0)
        del primals_66
        # Source Nodes: [x_238], Original ATen: [aten.convolution]
        buf356 = extern_kernels.convolution(buf355, primals_148, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf356, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf357 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_238], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf356, buf357, 15360, 64, grid=grid(15360, 64), stream=stream0)
        buf358 = buf350; del buf350  # reuse
        buf359 = buf349; del buf349  # reuse
        buf360 = buf348; del buf348  # reuse
        # Source Nodes: [x_239], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf357, buf358, buf359, buf360, 7680, 128, grid=grid(7680), stream=stream0)
        buf361 = buf352; del buf352  # reuse
        buf362 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf364 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_239], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf358, buf359, buf360, primals_274, primals_275, buf361, buf362, buf364, primals_274, primals_275, 1920, 4, grid=grid(1920), stream=stream0)
        del primals_274
        del primals_275
        buf365 = reinterpret_tensor(buf356, (8, 1920, 8, 8), (122880, 1, 15360, 1920), 0); del buf356  # reuse
        # Source Nodes: [x_239, x_243], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf357, buf361, buf362, primals_67, primals_68, buf365, 983040, grid=grid(983040), stream=stream0)
        del primals_68
        # Source Nodes: [x_246], Original ATen: [aten.convolution]
        buf366 = extern_kernels.convolution(buf365, primals_149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf366, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf367 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_246], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf366, buf367, 5120, 64, grid=grid(5120, 64), stream=stream0)
        buf368 = buf339; del buf339  # reuse
        buf369 = buf338; del buf338  # reuse
        buf370 = buf337; del buf337  # reuse
        # Source Nodes: [x_247], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf367, buf368, buf369, buf370, 2560, 128, grid=grid(2560), stream=stream0)
        buf371 = buf341; del buf341  # reuse
        buf372 = buf332; del buf332  # reuse
        buf374 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_247], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf368, buf369, buf370, primals_277, primals_278, buf371, buf372, buf374, primals_277, primals_278, 640, 4, grid=grid(640), stream=stream0)
        del primals_277
        del primals_278
        buf375 = reinterpret_tensor(buf366, (8, 640, 8, 8), (40960, 1, 5120, 640), 0); del buf366  # reuse
        # Source Nodes: [shortcut_11, x_247, x_254], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_49.run(buf367, buf371, buf372, primals_69, primals_70, buf345, buf375, 327680, grid=grid(327680), stream=stream0)
        del primals_70
        # Source Nodes: [x_255], Original ATen: [aten.convolution]
        buf376 = extern_kernels.convolution(buf375, primals_150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf377 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_255], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf376, buf377, 15360, 64, grid=grid(15360, 64), stream=stream0)
        buf378 = buf360; del buf360  # reuse
        buf379 = buf359; del buf359  # reuse
        buf380 = buf358; del buf358  # reuse
        # Source Nodes: [x_256], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf377, buf378, buf379, buf380, 7680, 128, grid=grid(7680), stream=stream0)
        buf381 = buf362; del buf362  # reuse
        buf382 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf384 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_256], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf378, buf379, buf380, primals_280, primals_281, buf381, buf382, buf384, primals_280, primals_281, 1920, 4, grid=grid(1920), stream=stream0)
        del primals_280
        del primals_281
        buf385 = reinterpret_tensor(buf376, (8, 1920, 8, 8), (122880, 1, 15360, 1920), 0); del buf376  # reuse
        # Source Nodes: [x_256, x_260], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf377, buf381, buf382, primals_71, primals_72, buf385, 983040, grid=grid(983040), stream=stream0)
        del primals_72
        # Source Nodes: [x_261], Original ATen: [aten.convolution]
        buf386 = extern_kernels.convolution(buf385, primals_151, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf386, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf387 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_261], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf386, buf387, 15360, 64, grid=grid(15360, 64), stream=stream0)
        buf388 = buf380; del buf380  # reuse
        buf389 = buf379; del buf379  # reuse
        buf390 = buf378; del buf378  # reuse
        # Source Nodes: [x_262], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf387, buf388, buf389, buf390, 7680, 128, grid=grid(7680), stream=stream0)
        buf391 = buf382; del buf382  # reuse
        buf392 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf394 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_262], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf388, buf389, buf390, primals_283, primals_284, buf391, buf392, buf394, primals_283, primals_284, 1920, 4, grid=grid(1920), stream=stream0)
        del primals_283
        del primals_284
        buf395 = reinterpret_tensor(buf386, (8, 1920, 8, 8), (122880, 1, 15360, 1920), 0); del buf386  # reuse
        # Source Nodes: [x_262, x_266], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf387, buf391, buf392, primals_73, primals_74, buf395, 983040, grid=grid(983040), stream=stream0)
        del primals_74
        # Source Nodes: [x_269], Original ATen: [aten.convolution]
        buf396 = extern_kernels.convolution(buf395, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf396, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf397 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_269], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf396, buf397, 5120, 64, grid=grid(5120, 64), stream=stream0)
        buf398 = buf370; del buf370  # reuse
        buf399 = buf369; del buf369  # reuse
        buf400 = buf368; del buf368  # reuse
        # Source Nodes: [x_270], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf397, buf398, buf399, buf400, 2560, 128, grid=grid(2560), stream=stream0)
        buf401 = buf372; del buf372  # reuse
        buf402 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf404 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_270], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf398, buf399, buf400, primals_286, primals_287, buf401, buf402, buf404, primals_286, primals_287, 640, 4, grid=grid(640), stream=stream0)
        del primals_286
        del primals_287
        buf405 = reinterpret_tensor(buf396, (8, 640, 8, 8), (40960, 1, 5120, 640), 0); del buf396  # reuse
        # Source Nodes: [shortcut_12, x_270, x_277], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_49.run(buf397, buf401, buf402, primals_75, primals_76, buf375, buf405, 327680, grid=grid(327680), stream=stream0)
        del primals_76
        # Source Nodes: [x_278], Original ATen: [aten.convolution]
        buf406 = extern_kernels.convolution(buf405, primals_153, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf406, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf407 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_278], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf406, buf407, 15360, 64, grid=grid(15360, 64), stream=stream0)
        buf408 = buf390; del buf390  # reuse
        buf409 = buf389; del buf389  # reuse
        buf410 = buf388; del buf388  # reuse
        # Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf407, buf408, buf409, buf410, 7680, 128, grid=grid(7680), stream=stream0)
        buf411 = buf392; del buf392  # reuse
        buf412 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf414 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf408, buf409, buf410, primals_289, primals_290, buf411, buf412, buf414, primals_289, primals_290, 1920, 4, grid=grid(1920), stream=stream0)
        del primals_289
        del primals_290
        buf415 = reinterpret_tensor(buf406, (8, 1920, 8, 8), (122880, 1, 15360, 1920), 0); del buf406  # reuse
        # Source Nodes: [x_279, x_283], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf407, buf411, buf412, primals_77, primals_78, buf415, 983040, grid=grid(983040), stream=stream0)
        del primals_78
        # Source Nodes: [x_284], Original ATen: [aten.convolution]
        buf416 = extern_kernels.convolution(buf415, primals_154, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf416, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf417 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_284], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf416, buf417, 15360, 64, grid=grid(15360, 64), stream=stream0)
        buf418 = buf410; del buf410  # reuse
        buf419 = buf409; del buf409  # reuse
        buf420 = buf408; del buf408  # reuse
        # Source Nodes: [x_285], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf417, buf418, buf419, buf420, 7680, 128, grid=grid(7680), stream=stream0)
        buf421 = buf412; del buf412  # reuse
        buf422 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf424 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_285], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf418, buf419, buf420, primals_292, primals_293, buf421, buf422, buf424, primals_292, primals_293, 1920, 4, grid=grid(1920), stream=stream0)
        del primals_292
        del primals_293
        buf425 = reinterpret_tensor(buf416, (8, 1920, 8, 8), (122880, 1, 15360, 1920), 0); del buf416  # reuse
        # Source Nodes: [x_285, x_289], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf417, buf421, buf422, primals_79, primals_80, buf425, 983040, grid=grid(983040), stream=stream0)
        del primals_80
        # Source Nodes: [x_292], Original ATen: [aten.convolution]
        buf426 = extern_kernels.convolution(buf425, primals_155, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf426, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf427 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_292], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf426, buf427, 5120, 64, grid=grid(5120, 64), stream=stream0)
        buf428 = buf400; del buf400  # reuse
        buf429 = buf399; del buf399  # reuse
        buf430 = buf398; del buf398  # reuse
        # Source Nodes: [x_293], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf427, buf428, buf429, buf430, 2560, 128, grid=grid(2560), stream=stream0)
        buf431 = buf402; del buf402  # reuse
        buf432 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf434 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_293], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf428, buf429, buf430, primals_295, primals_296, buf431, buf432, buf434, primals_295, primals_296, 640, 4, grid=grid(640), stream=stream0)
        del primals_295
        del primals_296
        buf435 = reinterpret_tensor(buf426, (8, 640, 8, 8), (40960, 1, 5120, 640), 0); del buf426  # reuse
        # Source Nodes: [shortcut_13, x_293, x_300], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_49.run(buf427, buf431, buf432, primals_81, primals_82, buf405, buf435, 327680, grid=grid(327680), stream=stream0)
        del primals_82
        # Source Nodes: [x_301], Original ATen: [aten.convolution]
        buf436 = extern_kernels.convolution(buf435, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf436, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf437 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_301], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf436, buf437, 15360, 64, grid=grid(15360, 64), stream=stream0)
        buf438 = buf420; del buf420  # reuse
        buf439 = buf419; del buf419  # reuse
        buf440 = buf418; del buf418  # reuse
        # Source Nodes: [x_302], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf437, buf438, buf439, buf440, 7680, 128, grid=grid(7680), stream=stream0)
        buf441 = buf422; del buf422  # reuse
        buf442 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf444 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_302], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf438, buf439, buf440, primals_298, primals_299, buf441, buf442, buf444, primals_298, primals_299, 1920, 4, grid=grid(1920), stream=stream0)
        del primals_298
        del primals_299
        buf445 = reinterpret_tensor(buf436, (8, 1920, 8, 8), (122880, 1, 15360, 1920), 0); del buf436  # reuse
        # Source Nodes: [x_302, x_306], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf437, buf441, buf442, primals_83, primals_84, buf445, 983040, grid=grid(983040), stream=stream0)
        del primals_84
        # Source Nodes: [x_307], Original ATen: [aten.convolution]
        buf446 = extern_kernels.convolution(buf445, primals_157, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf446, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf447 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_307], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf446, buf447, 15360, 64, grid=grid(15360, 64), stream=stream0)
        buf448 = buf440; del buf440  # reuse
        buf449 = buf439; del buf439  # reuse
        buf450 = buf438; del buf438  # reuse
        # Source Nodes: [x_308], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf447, buf448, buf449, buf450, 7680, 128, grid=grid(7680), stream=stream0)
        buf451 = buf442; del buf442  # reuse
        buf452 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf454 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_308], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf448, buf449, buf450, primals_301, primals_302, buf451, buf452, buf454, primals_301, primals_302, 1920, 4, grid=grid(1920), stream=stream0)
        del primals_301
        del primals_302
        buf455 = reinterpret_tensor(buf446, (8, 1920, 8, 8), (122880, 1, 15360, 1920), 0); del buf446  # reuse
        # Source Nodes: [x_308, x_312], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf447, buf451, buf452, primals_85, primals_86, buf455, 983040, grid=grid(983040), stream=stream0)
        del primals_86
        # Source Nodes: [x_315], Original ATen: [aten.convolution]
        buf456 = extern_kernels.convolution(buf455, primals_158, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf456, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf457 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_315], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf456, buf457, 5120, 64, grid=grid(5120, 64), stream=stream0)
        buf458 = buf430; del buf430  # reuse
        buf459 = buf429; del buf429  # reuse
        buf460 = buf428; del buf428  # reuse
        # Source Nodes: [x_316], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf457, buf458, buf459, buf460, 2560, 128, grid=grid(2560), stream=stream0)
        buf461 = buf432; del buf432  # reuse
        buf462 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf464 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_316], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf458, buf459, buf460, primals_304, primals_305, buf461, buf462, buf464, primals_304, primals_305, 640, 4, grid=grid(640), stream=stream0)
        del primals_304
        del primals_305
        buf465 = reinterpret_tensor(buf456, (8, 640, 8, 8), (40960, 1, 5120, 640), 0); del buf456  # reuse
        # Source Nodes: [shortcut_14, x_316, x_323], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_49.run(buf457, buf461, buf462, primals_87, primals_88, buf435, buf465, 327680, grid=grid(327680), stream=stream0)
        del primals_88
        # Source Nodes: [x_324], Original ATen: [aten.convolution]
        buf466 = extern_kernels.convolution(buf465, primals_159, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf466, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf467 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_324], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf466, buf467, 15360, 64, grid=grid(15360, 64), stream=stream0)
        buf468 = buf450; del buf450  # reuse
        buf469 = buf449; del buf449  # reuse
        buf470 = buf448; del buf448  # reuse
        # Source Nodes: [x_325], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf467, buf468, buf469, buf470, 7680, 128, grid=grid(7680), stream=stream0)
        buf471 = buf452; del buf452  # reuse
        buf472 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf474 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_325], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf468, buf469, buf470, primals_307, primals_308, buf471, buf472, buf474, primals_307, primals_308, 1920, 4, grid=grid(1920), stream=stream0)
        del primals_307
        del primals_308
        buf475 = reinterpret_tensor(buf466, (8, 1920, 8, 8), (122880, 1, 15360, 1920), 0); del buf466  # reuse
        # Source Nodes: [x_325, x_329], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf467, buf471, buf472, primals_89, primals_90, buf475, 983040, grid=grid(983040), stream=stream0)
        del primals_90
        # Source Nodes: [x_330], Original ATen: [aten.convolution]
        buf476 = extern_kernels.convolution(buf475, primals_160, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf476, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf477 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_330], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf476, buf477, 15360, 64, grid=grid(15360, 64), stream=stream0)
        buf478 = buf470; del buf470  # reuse
        buf479 = buf469; del buf469  # reuse
        buf480 = buf468; del buf468  # reuse
        # Source Nodes: [x_331], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf477, buf478, buf479, buf480, 7680, 128, grid=grid(7680), stream=stream0)
        buf481 = buf472; del buf472  # reuse
        buf482 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf484 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_331], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf478, buf479, buf480, primals_310, primals_311, buf481, buf482, buf484, primals_310, primals_311, 1920, 4, grid=grid(1920), stream=stream0)
        del primals_310
        del primals_311
        buf485 = reinterpret_tensor(buf476, (8, 1920, 8, 8), (122880, 1, 15360, 1920), 0); del buf476  # reuse
        # Source Nodes: [x_331, x_335], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf477, buf481, buf482, primals_91, primals_92, buf485, 983040, grid=grid(983040), stream=stream0)
        del primals_92
        # Source Nodes: [x_338], Original ATen: [aten.convolution]
        buf486 = extern_kernels.convolution(buf485, primals_161, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf486, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf487 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_338], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf486, buf487, 5120, 64, grid=grid(5120, 64), stream=stream0)
        buf488 = buf460; del buf460  # reuse
        buf489 = buf459; del buf459  # reuse
        buf490 = buf458; del buf458  # reuse
        # Source Nodes: [x_339], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf487, buf488, buf489, buf490, 2560, 128, grid=grid(2560), stream=stream0)
        buf491 = buf462; del buf462  # reuse
        buf492 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf494 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_339], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf488, buf489, buf490, primals_313, primals_314, buf491, buf492, buf494, primals_313, primals_314, 640, 4, grid=grid(640), stream=stream0)
        del primals_313
        del primals_314
        buf495 = reinterpret_tensor(buf486, (8, 640, 8, 8), (40960, 1, 5120, 640), 0); del buf486  # reuse
        # Source Nodes: [shortcut_15, x_339, x_346], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_49.run(buf487, buf491, buf492, primals_93, primals_94, buf465, buf495, 327680, grid=grid(327680), stream=stream0)
        del primals_94
        # Source Nodes: [x_347], Original ATen: [aten.convolution]
        buf496 = extern_kernels.convolution(buf495, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf496, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf497 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_347], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf496, buf497, 15360, 64, grid=grid(15360, 64), stream=stream0)
        buf498 = buf480; del buf480  # reuse
        buf499 = buf479; del buf479  # reuse
        buf500 = buf478; del buf478  # reuse
        # Source Nodes: [x_348], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf497, buf498, buf499, buf500, 7680, 128, grid=grid(7680), stream=stream0)
        buf501 = buf482; del buf482  # reuse
        buf502 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf504 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_348], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf498, buf499, buf500, primals_316, primals_317, buf501, buf502, buf504, primals_316, primals_317, 1920, 4, grid=grid(1920), stream=stream0)
        del primals_316
        del primals_317
        buf505 = reinterpret_tensor(buf496, (8, 1920, 8, 8), (122880, 1, 15360, 1920), 0); del buf496  # reuse
        # Source Nodes: [x_348, x_352], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf497, buf501, buf502, primals_95, primals_96, buf505, 983040, grid=grid(983040), stream=stream0)
        del primals_96
        # Source Nodes: [x_353], Original ATen: [aten.convolution]
        buf506 = extern_kernels.convolution(buf505, primals_163, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf506, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf507 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_353], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf506, buf507, 15360, 64, grid=grid(15360, 64), stream=stream0)
        buf508 = buf500; del buf500  # reuse
        buf509 = buf499; del buf499  # reuse
        buf510 = buf498; del buf498  # reuse
        # Source Nodes: [x_354], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf507, buf508, buf509, buf510, 7680, 128, grid=grid(7680), stream=stream0)
        buf511 = buf502; del buf502  # reuse
        buf512 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf514 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_354], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf508, buf509, buf510, primals_319, primals_320, buf511, buf512, buf514, primals_319, primals_320, 1920, 4, grid=grid(1920), stream=stream0)
        del primals_319
        del primals_320
        buf515 = reinterpret_tensor(buf506, (8, 1920, 8, 8), (122880, 1, 15360, 1920), 0); del buf506  # reuse
        # Source Nodes: [x_354, x_358], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf507, buf511, buf512, primals_97, primals_98, buf515, 983040, grid=grid(983040), stream=stream0)
        del primals_98
        # Source Nodes: [x_361], Original ATen: [aten.convolution]
        buf516 = extern_kernels.convolution(buf515, primals_164, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf516, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf517 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_361], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf516, buf517, 5120, 64, grid=grid(5120, 64), stream=stream0)
        buf518 = buf490; del buf490  # reuse
        buf519 = buf489; del buf489  # reuse
        buf520 = buf488; del buf488  # reuse
        # Source Nodes: [x_362], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf517, buf518, buf519, buf520, 2560, 128, grid=grid(2560), stream=stream0)
        buf521 = buf492; del buf492  # reuse
        buf522 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf524 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_362], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf518, buf519, buf520, primals_322, primals_323, buf521, buf522, buf524, primals_322, primals_323, 640, 4, grid=grid(640), stream=stream0)
        del primals_322
        del primals_323
        buf525 = reinterpret_tensor(buf516, (8, 640, 8, 8), (40960, 1, 5120, 640), 0); del buf516  # reuse
        # Source Nodes: [shortcut_16, x_362, x_369], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_49.run(buf517, buf521, buf522, primals_99, primals_100, buf495, buf525, 327680, grid=grid(327680), stream=stream0)
        del primals_100
        # Source Nodes: [x_370], Original ATen: [aten.convolution]
        buf526 = extern_kernels.convolution(buf525, primals_165, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf526, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf527 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_370], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf526, buf527, 15360, 64, grid=grid(15360, 64), stream=stream0)
        buf528 = buf510; del buf510  # reuse
        buf529 = buf509; del buf509  # reuse
        buf530 = buf508; del buf508  # reuse
        # Source Nodes: [x_371], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf527, buf528, buf529, buf530, 7680, 128, grid=grid(7680), stream=stream0)
        buf531 = buf512; del buf512  # reuse
        buf532 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf534 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_371], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf528, buf529, buf530, primals_325, primals_326, buf531, buf532, buf534, primals_325, primals_326, 1920, 4, grid=grid(1920), stream=stream0)
        del primals_325
        del primals_326
        buf535 = reinterpret_tensor(buf526, (8, 1920, 8, 8), (122880, 1, 15360, 1920), 0); del buf526  # reuse
        # Source Nodes: [x_371, x_375], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf527, buf531, buf532, primals_101, primals_102, buf535, 983040, grid=grid(983040), stream=stream0)
        del primals_102
        # Source Nodes: [x_376], Original ATen: [aten.convolution]
        buf536 = extern_kernels.convolution(buf535, primals_166, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf536, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf537 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_376], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf536, buf537, 15360, 64, grid=grid(15360, 64), stream=stream0)
        buf538 = buf530; del buf530  # reuse
        buf539 = buf529; del buf529  # reuse
        buf540 = buf528; del buf528  # reuse
        # Source Nodes: [x_377], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf537, buf538, buf539, buf540, 7680, 128, grid=grid(7680), stream=stream0)
        buf541 = buf532; del buf532  # reuse
        buf542 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf544 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_377], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf538, buf539, buf540, primals_328, primals_329, buf541, buf542, buf544, primals_328, primals_329, 1920, 4, grid=grid(1920), stream=stream0)
        del primals_328
        del primals_329
        buf545 = reinterpret_tensor(buf536, (8, 1920, 8, 8), (122880, 1, 15360, 1920), 0); del buf536  # reuse
        # Source Nodes: [x_377, x_381], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf537, buf541, buf542, primals_103, primals_104, buf545, 983040, grid=grid(983040), stream=stream0)
        del primals_104
        # Source Nodes: [x_384], Original ATen: [aten.convolution]
        buf546 = extern_kernels.convolution(buf545, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf546, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf547 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_384], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf546, buf547, 5120, 64, grid=grid(5120, 64), stream=stream0)
        buf548 = buf520; del buf520  # reuse
        buf549 = buf519; del buf519  # reuse
        buf550 = buf518; del buf518  # reuse
        # Source Nodes: [x_385], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf547, buf548, buf549, buf550, 2560, 128, grid=grid(2560), stream=stream0)
        buf551 = buf522; del buf522  # reuse
        buf552 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf554 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_385], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf548, buf549, buf550, primals_331, primals_332, buf551, buf552, buf554, primals_331, primals_332, 640, 4, grid=grid(640), stream=stream0)
        del primals_331
        del primals_332
        buf555 = reinterpret_tensor(buf546, (8, 640, 8, 8), (40960, 1, 5120, 640), 0); del buf546  # reuse
        # Source Nodes: [shortcut_17, x_385, x_392], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_49.run(buf547, buf551, buf552, primals_105, primals_106, buf525, buf555, 327680, grid=grid(327680), stream=stream0)
        del primals_106
        # Source Nodes: [x_393], Original ATen: [aten.convolution]
        buf556 = extern_kernels.convolution(buf555, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf556, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf557 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_393], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf556, buf557, 15360, 64, grid=grid(15360, 64), stream=stream0)
        buf558 = buf540; del buf540  # reuse
        buf559 = buf539; del buf539  # reuse
        buf560 = buf538; del buf538  # reuse
        # Source Nodes: [x_394], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf557, buf558, buf559, buf560, 7680, 128, grid=grid(7680), stream=stream0)
        buf561 = buf542; del buf542  # reuse
        buf562 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf564 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_394], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf558, buf559, buf560, primals_334, primals_335, buf561, buf562, buf564, primals_334, primals_335, 1920, 4, grid=grid(1920), stream=stream0)
        del primals_334
        del primals_335
        buf565 = reinterpret_tensor(buf556, (8, 1920, 8, 8), (122880, 1, 15360, 1920), 0); del buf556  # reuse
        # Source Nodes: [x_394, x_398], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf557, buf561, buf562, primals_107, primals_108, buf565, 983040, grid=grid(983040), stream=stream0)
        del primals_108
        # Source Nodes: [x_399], Original ATen: [aten.convolution]
        buf566 = extern_kernels.convolution(buf565, primals_169, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf566, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf567 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_399], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf566, buf567, 15360, 64, grid=grid(15360, 64), stream=stream0)
        buf568 = buf560; del buf560  # reuse
        buf569 = buf559; del buf559  # reuse
        buf570 = buf558; del buf558  # reuse
        # Source Nodes: [x_400], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf567, buf568, buf569, buf570, 7680, 128, grid=grid(7680), stream=stream0)
        buf571 = buf562; del buf562  # reuse
        buf572 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf574 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_400], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf568, buf569, buf570, primals_337, primals_338, buf571, buf572, buf574, primals_337, primals_338, 1920, 4, grid=grid(1920), stream=stream0)
        del buf568
        del buf569
        del buf570
        del primals_337
        del primals_338
        buf575 = reinterpret_tensor(buf566, (8, 1920, 8, 8), (122880, 1, 15360, 1920), 0); del buf566  # reuse
        # Source Nodes: [x_400, x_404], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf567, buf571, buf572, primals_109, primals_110, buf575, 983040, grid=grid(983040), stream=stream0)
        del buf572
        del primals_110
        # Source Nodes: [x_407], Original ATen: [aten.convolution]
        buf576 = extern_kernels.convolution(buf575, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf576, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf577 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_407], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf576, buf577, 5120, 64, grid=grid(5120, 64), stream=stream0)
        buf578 = buf550; del buf550  # reuse
        buf579 = buf549; del buf549  # reuse
        buf580 = buf548; del buf548  # reuse
        # Source Nodes: [x_408], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf577, buf578, buf579, buf580, 2560, 128, grid=grid(2560), stream=stream0)
        buf581 = buf552; del buf552  # reuse
        buf582 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf584 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_408], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf578, buf579, buf580, primals_340, primals_341, buf581, buf582, buf584, primals_340, primals_341, 640, 4, grid=grid(640), stream=stream0)
        del primals_340
        del primals_341
        buf585 = reinterpret_tensor(buf576, (8, 640, 8, 8), (40960, 1, 5120, 640), 0); del buf576  # reuse
        # Source Nodes: [x_408, x_415, x_416], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_49.run(buf577, buf581, buf582, primals_111, primals_112, buf555, buf585, 327680, grid=grid(327680), stream=stream0)
        del buf582
        del primals_112
        # Source Nodes: [x_417], Original ATen: [aten.convolution]
        buf586 = extern_kernels.convolution(buf585, primals_171, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf586, (8, 2560, 8, 8), (163840, 64, 8, 1))
        buf587 = empty_strided((8, 2560, 8, 8), (163840, 1, 20480, 2560), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_417], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(buf586, buf587, 20480, 64, grid=grid(20480, 64), stream=stream0)
        buf588 = reinterpret_tensor(buf300, (1, 2560, 1, 1, 4), (10240, 1, 10240, 10240, 2560), 0); del buf300  # reuse
        buf589 = reinterpret_tensor(buf299, (1, 2560, 1, 1, 4), (10240, 1, 10240, 10240, 2560), 0); del buf299  # reuse
        buf590 = reinterpret_tensor(buf298, (1, 2560, 1, 1, 4), (10240, 1, 10240, 10240, 2560), 0); del buf298  # reuse
        # Source Nodes: [x_418], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf587, buf588, buf589, buf590, 10240, 128, grid=grid(10240), stream=stream0)
        buf591 = reinterpret_tensor(buf580, (1, 2560, 1, 1), (2560, 1, 2560, 2560), 0); del buf580  # reuse
        buf592 = reinterpret_tensor(buf579, (1, 2560, 1, 1), (2560, 1, 2560, 2560), 0); del buf579  # reuse
        buf594 = reinterpret_tensor(buf578, (2560, ), (1, ), 0); del buf578  # reuse
        # Source Nodes: [x_418], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_52.run(buf588, buf589, buf590, primals_343, primals_344, buf591, buf592, buf594, primals_343, primals_344, 2560, 4, grid=grid(2560), stream=stream0)
        del buf588
        del buf589
        del buf590
        del primals_343
        del primals_344
        buf595 = reinterpret_tensor(buf586, (8, 2560, 8, 8), (163840, 1, 20480, 2560), 0); del buf586  # reuse
        buf599 = empty_strided((8, 2560, 8, 8), (163840, 1, 20480, 2560), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_418, x_423], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_53.run(buf587, buf591, buf592, primals_113, primals_114, buf595, buf599, 1310720, grid=grid(1310720), stream=stream0)
        del buf592
        del primals_114
        buf596 = empty_strided((8, 2560, 1, 1), (2560, 1, 20480, 20480), device='cuda', dtype=torch.float32)
        buf597 = reinterpret_tensor(buf596, (8, 2560), (2560, 1), 0); del buf596  # reuse
        # Source Nodes: [x_424, x_426], Original ATen: [aten.mean, aten.view]
        triton_per_fused_mean_view_54.run(buf597, buf595, 20480, 64, grid=grid(20480), stream=stream0)
        del buf595
        buf598 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_428], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_173, buf597, reinterpret_tensor(primals_172, (2560, 1000), (1, 2560), 0), alpha=1, beta=1, out=buf598)
        del primals_173
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_174, primals_174, 1, grid=grid(1), stream=stream0)
        del primals_174
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_177, primals_177, 1, grid=grid(1), stream=stream0)
        del primals_177
        # Source Nodes: [add__2], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_180, primals_180, 1, grid=grid(1), stream=stream0)
        del primals_180
        # Source Nodes: [add__3], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_183, primals_183, 1, grid=grid(1), stream=stream0)
        del primals_183
        # Source Nodes: [add__4], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_186, primals_186, 1, grid=grid(1), stream=stream0)
        del primals_186
        # Source Nodes: [add__5], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_189, primals_189, 1, grid=grid(1), stream=stream0)
        del primals_189
        # Source Nodes: [add__6], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_192, primals_192, 1, grid=grid(1), stream=stream0)
        del primals_192
        # Source Nodes: [add__7], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_195, primals_195, 1, grid=grid(1), stream=stream0)
        del primals_195
        # Source Nodes: [add__8], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_198, primals_198, 1, grid=grid(1), stream=stream0)
        del primals_198
        # Source Nodes: [add__9], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_201, primals_201, 1, grid=grid(1), stream=stream0)
        del primals_201
        # Source Nodes: [add__10], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_204, primals_204, 1, grid=grid(1), stream=stream0)
        del primals_204
        # Source Nodes: [add__11], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_207, primals_207, 1, grid=grid(1), stream=stream0)
        del primals_207
        # Source Nodes: [add__12], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_210, primals_210, 1, grid=grid(1), stream=stream0)
        del primals_210
        # Source Nodes: [add__13], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_213, primals_213, 1, grid=grid(1), stream=stream0)
        del primals_213
        # Source Nodes: [add__14], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_216, primals_216, 1, grid=grid(1), stream=stream0)
        del primals_216
        # Source Nodes: [add__15], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_219, primals_219, 1, grid=grid(1), stream=stream0)
        del primals_219
        # Source Nodes: [add__16], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_222, primals_222, 1, grid=grid(1), stream=stream0)
        del primals_222
        # Source Nodes: [add__17], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_225, primals_225, 1, grid=grid(1), stream=stream0)
        del primals_225
        # Source Nodes: [add__18], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_228, primals_228, 1, grid=grid(1), stream=stream0)
        del primals_228
        # Source Nodes: [add__19], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_231, primals_231, 1, grid=grid(1), stream=stream0)
        del primals_231
        # Source Nodes: [add__20], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_234, primals_234, 1, grid=grid(1), stream=stream0)
        del primals_234
        # Source Nodes: [add__21], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_237, primals_237, 1, grid=grid(1), stream=stream0)
        del primals_237
        # Source Nodes: [add__22], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_240, primals_240, 1, grid=grid(1), stream=stream0)
        del primals_240
        # Source Nodes: [add__23], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_243, primals_243, 1, grid=grid(1), stream=stream0)
        del primals_243
        # Source Nodes: [add__24], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_246, primals_246, 1, grid=grid(1), stream=stream0)
        del primals_246
        # Source Nodes: [add__25], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_249, primals_249, 1, grid=grid(1), stream=stream0)
        del primals_249
        # Source Nodes: [add__26], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_252, primals_252, 1, grid=grid(1), stream=stream0)
        del primals_252
        # Source Nodes: [add__27], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_255, primals_255, 1, grid=grid(1), stream=stream0)
        del primals_255
        # Source Nodes: [add__28], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_258, primals_258, 1, grid=grid(1), stream=stream0)
        del primals_258
        # Source Nodes: [add__29], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_261, primals_261, 1, grid=grid(1), stream=stream0)
        del primals_261
        # Source Nodes: [add__30], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_264, primals_264, 1, grid=grid(1), stream=stream0)
        del primals_264
        # Source Nodes: [add__31], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_267, primals_267, 1, grid=grid(1), stream=stream0)
        del primals_267
        # Source Nodes: [add__32], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_270, primals_270, 1, grid=grid(1), stream=stream0)
        del primals_270
        # Source Nodes: [add__33], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_273, primals_273, 1, grid=grid(1), stream=stream0)
        del primals_273
        # Source Nodes: [add__34], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_276, primals_276, 1, grid=grid(1), stream=stream0)
        del primals_276
        # Source Nodes: [add__35], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_279, primals_279, 1, grid=grid(1), stream=stream0)
        del primals_279
        # Source Nodes: [add__36], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_282, primals_282, 1, grid=grid(1), stream=stream0)
        del primals_282
        # Source Nodes: [add__37], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_285, primals_285, 1, grid=grid(1), stream=stream0)
        del primals_285
        # Source Nodes: [add__38], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_288, primals_288, 1, grid=grid(1), stream=stream0)
        del primals_288
        # Source Nodes: [add__39], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_291, primals_291, 1, grid=grid(1), stream=stream0)
        del primals_291
        # Source Nodes: [add__40], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_294, primals_294, 1, grid=grid(1), stream=stream0)
        del primals_294
        # Source Nodes: [add__41], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_297, primals_297, 1, grid=grid(1), stream=stream0)
        del primals_297
        # Source Nodes: [add__42], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_300, primals_300, 1, grid=grid(1), stream=stream0)
        del primals_300
        # Source Nodes: [add__43], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_303, primals_303, 1, grid=grid(1), stream=stream0)
        del primals_303
        # Source Nodes: [add__44], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_306, primals_306, 1, grid=grid(1), stream=stream0)
        del primals_306
        # Source Nodes: [add__45], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_309, primals_309, 1, grid=grid(1), stream=stream0)
        del primals_309
        # Source Nodes: [add__46], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_312, primals_312, 1, grid=grid(1), stream=stream0)
        del primals_312
        # Source Nodes: [add__47], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_315, primals_315, 1, grid=grid(1), stream=stream0)
        del primals_315
        # Source Nodes: [add__48], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_318, primals_318, 1, grid=grid(1), stream=stream0)
        del primals_318
        # Source Nodes: [add__49], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_321, primals_321, 1, grid=grid(1), stream=stream0)
        del primals_321
        # Source Nodes: [add__50], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_324, primals_324, 1, grid=grid(1), stream=stream0)
        del primals_324
        # Source Nodes: [add__51], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_327, primals_327, 1, grid=grid(1), stream=stream0)
        del primals_327
        # Source Nodes: [add__52], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_330, primals_330, 1, grid=grid(1), stream=stream0)
        del primals_330
        # Source Nodes: [add__53], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_333, primals_333, 1, grid=grid(1), stream=stream0)
        del primals_333
        # Source Nodes: [add__54], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_336, primals_336, 1, grid=grid(1), stream=stream0)
        del primals_336
        # Source Nodes: [add__55], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_339, primals_339, 1, grid=grid(1), stream=stream0)
        del primals_339
        # Source Nodes: [add__56], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_342, primals_342, 1, grid=grid(1), stream=stream0)
        del primals_342
        return (buf598, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, buf0, buf1, buf2, primals_118, buf3, buf4, primals_121, buf5, buf6, primals_124, buf7, primals_126, primals_127, primals_128, buf8, primals_130, primals_131, buf9, primals_133, primals_134, buf10, primals_136, primals_137, buf11, primals_139, primals_140, buf12, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, buf13, buf15, buf25, buf26, buf28, buf38, buf39, buf41, buf51, buf53, buf63, buf65, buf67, buf74, buf75, buf77, buf84, buf86, buf93, buf95, buf97, buf104, buf105, buf107, buf114, buf115, buf117, buf124, buf125, buf127, buf134, buf135, buf137, buf144, buf146, buf153, buf155, buf157, buf164, buf165, buf167, buf174, buf175, buf177, buf184, buf185, buf187, buf194, buf195, buf197, buf204, buf205, buf207, buf214, buf215, buf217, buf224, buf225, buf227, buf234, buf235, buf237, buf244, buf245, buf247, buf254, buf255, buf257, buf264, buf265, buf267, buf274, buf275, buf277, buf284, buf285, buf287, buf294, buf295, buf297, buf304, buf305, buf307, buf314, buf315, buf317, buf324, buf325, buf327, buf334, buf336, buf343, buf345, buf347, buf354, buf355, buf357, buf364, buf365, buf367, buf374, buf375, buf377, buf384, buf385, buf387, buf394, buf395, buf397, buf404, buf405, buf407, buf414, buf415, buf417, buf424, buf425, buf427, buf434, buf435, buf437, buf444, buf445, buf447, buf454, buf455, buf457, buf464, buf465, buf467, buf474, buf475, buf477, buf484, buf485, buf487, buf494, buf495, buf497, buf504, buf505, buf507, buf514, buf515, buf517, buf524, buf525, buf527, buf534, buf535, buf537, buf544, buf545, buf547, buf554, buf555, buf557, buf564, buf565, buf567, buf574, buf575, buf577, buf584, buf585, buf587, buf594, buf597, reinterpret_tensor(primals_172, (1000, 2560), (2560, 1), 0), buf599, reinterpret_tensor(buf591, (1, 2560, 1, 1), (2560, 1, 1, 1), 0), reinterpret_tensor(buf581, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf571, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf561, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf551, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf541, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf531, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf521, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf511, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf501, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf491, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf481, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf471, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf461, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf451, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf441, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf431, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf421, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf411, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf401, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf391, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf381, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf371, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf361, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf351, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf340, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf331, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf321, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf311, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf301, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf291, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf281, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf271, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf261, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf251, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf241, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf231, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf221, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf211, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf201, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf191, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf181, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf171, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf161, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf150, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf141, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf131, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf121, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf111, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf101, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf90, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf81, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf71, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf60, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf48, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf35, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf22, (1, 32, 1, 1), (32, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((192, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((192, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((160, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((640, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((640, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((2560, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((1000, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_175 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_178 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_181 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_184 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_187 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_190 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_193 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_196 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_199 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_202 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_205 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_208 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_211 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_214 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_217 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_220 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_223 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_226 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_229 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_232 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_235 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_238 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_241 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_244 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_247 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_250 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_253 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_256 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_259 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_262 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_265 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_268 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_271 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_274 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_277 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_280 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_283 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_286 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_289 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_292 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_295 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_298 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_301 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_304 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_307 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_310 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_313 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_316 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_319 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_322 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_325 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_328 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_331 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_334 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_337 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_340 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_343 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gernet_l', benchmark_compiled_module)
